import { spawn } from 'node:child_process'
import { existsSync, readdirSync, statSync } from 'node:fs'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { extname, join } from 'node:path'

export const DEFAULT_GEMINI_IMAGE_MODEL = 'gemini-3.1-flash-image'

export const GEMINI_IMAGE_MODELS = [
  'gemini-3-pro-image',
  'gemini-3.1-flash-image',
  'gemini-2.5-flash-image',
] as const

export type GeminiImageModel = (typeof GEMINI_IMAGE_MODELS)[number]

export const LOCAL_GPT_IMAGE_MODEL = 'local/gpt-image-2'

export type ImageModel = GeminiImageModel | typeof LOCAL_GPT_IMAGE_MODEL

export type GeminiTextPart = {
  text: string
}

export type GeminiInlinePart = {
  inline_data: {
    mime_type: string
    data: string
  }
}

export type GeminiPart = GeminiTextPart | GeminiInlinePart

export type GeminiGenerationConfig = {
  temperature?: number
  maxOutputTokens?: number
}

export type GeminiCallOptions = {
  apiKey?: unknown
  model?: unknown
  parts: GeminiPart[]
  generationConfig?: GeminiGenerationConfig
}

export type ImageCallOptions = GeminiCallOptions & {
  outputWidth?: number
  outputHeight?: number
}

export class GeminiMissingApiKeyError extends Error {
  constructor() {
    super('Gemini API key missing. Add one in Settings.')
    this.name = 'GeminiMissingApiKeyError'
  }
}

export class GeminiApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'GeminiApiError'
    this.status = status
  }
}

export class LocalGptImageError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'LocalGptImageError'
  }
}

export function resolveGeminiApiKey(apiKey: unknown): string | undefined {
  if (typeof apiKey === 'string' && apiKey.trim()) {
    return apiKey.trim()
  }
  return process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || undefined
}

export function resolveGeminiModel(model: unknown): GeminiImageModel {
  if (
    typeof model === 'string' &&
    GEMINI_IMAGE_MODELS.includes(model.trim() as GeminiImageModel)
  ) {
    return model.trim() as GeminiImageModel
  }
  return DEFAULT_GEMINI_IMAGE_MODEL
}

export function isLocalGptImageModel(
  model: unknown
): model is typeof LOCAL_GPT_IMAGE_MODEL {
  return typeof model === 'string' && model.trim() === LOCAL_GPT_IMAGE_MODEL
}

export function resolveImageModel(model: unknown): ImageModel {
  if (isLocalGptImageModel(model)) return LOCAL_GPT_IMAGE_MODEL
  return resolveGeminiModel(model)
}

export function dataUrlToGeminiPart(dataUrl: string): GeminiInlinePart {
  const trimmed = dataUrl.trim()
  const match = trimmed.match(/^data:([^;,]+)?(?:;[^,]*)?;base64,([\s\S]*)$/i)

  if (!match) {
    return {
      inline_data: {
        mime_type: 'image/png',
        data: trimmed,
      },
    }
  }

  return {
    inline_data: {
      mime_type: match[1] || 'image/png',
      data: match[2],
    },
  }
}

export function imageGenerationConfig(
  aspectRatio?: string,
  model?: string
): GeminiGenerationConfig {
  void aspectRatio
  void model
  // The Gemini REST endpoint rejects SDK-style image response fields such as
  // responseFormat and responseModalities in generationConfig. Keep image sizing
  // guidance in the prompt text instead of sending unsupported schema fields.
  return {}
}

export async function callGeminiGenerateContent(
  options: GeminiCallOptions
): Promise<unknown> {
  const key = resolveGeminiApiKey(options.apiKey)
  if (!key) throw new GeminiMissingApiKeyError()

  const model = resolveGeminiModel(options.model)
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1/models/${encodeURIComponent(model)}:generateContent`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': key,
      },
      body: JSON.stringify({
        contents: [
          {
            role: 'user',
            parts: options.parts,
          },
        ],
        ...(options.generationConfig
          ? { generationConfig: options.generationConfig }
          : {}),
      }),
    }
  )

  if (!response.ok) {
    throw new GeminiApiError(await geminiErrorMessage(response), response.status)
  }

  return response.json()
}

export async function callImageGenerateContent(
  options: ImageCallOptions
): Promise<unknown> {
  const model = resolveImageModel(options.model)
  if (isLocalGptImageModel(model)) {
    return callLocalGptImageGenerateContent({ ...options, model })
  }

  return callGeminiGenerateContent({ ...options, model })
}

async function callLocalGptImageGenerateContent(
  options: ImageCallOptions & { model: typeof LOCAL_GPT_IMAGE_MODEL }
): Promise<unknown> {
  const runDir = await mkdtemp(join(tmpdir(), 'image-extender-local-gpt-'))
  const messagePath = join(runDir, 'last-message.txt')
  try {
    const imagePaths = await writeLocalGptInputImages(options.parts, runDir)
    const prompt = buildLocalGptPrompt(options)
    const command = resolveCodexCommand()
    const generatedImagesBefore = snapshotGeneratedImages()
    const args = [
      ...command.argsPrefix,
      'exec',
      '--json',
      '--sandbox',
      'danger-full-access',
      '--skip-git-repo-check',
      '-C',
      process.cwd(),
      ...imagePaths.flatMap((imagePath) => ['--image', imagePath]),
      '-o',
      messagePath,
      prompt,
    ]
    const result = await runCommand(command.command, args, process.cwd())
    const lastMessage = existsSync(messagePath)
      ? await readFile(messagePath, 'utf8')
      : ''

    if (result.exitCode !== 0) {
      throw new LocalGptImageError(
        [
          `Local GPT image generation failed with exit code ${result.exitCode}.`,
          result.stderr.trim(),
          lastMessage.trim(),
        ]
          .filter(Boolean)
          .join('\n')
      )
    }

    const generatedImage = findNewestGeneratedImage(generatedImagesBefore)
    if (!generatedImage) {
      throw new LocalGptImageError(
        [
          'Local GPT did not create an image file.',
          lastMessage.trim(),
          result.stderr.trim(),
        ]
          .filter(Boolean)
          .join('\n')
      )
    }

    const imageBuffer = await readFile(generatedImage.path)
    const mimeType = mimeTypeForFile(generatedImage.path)
    return {
      candidates: [
        {
          content: {
            parts: [
              {
                inlineData: {
                  mimeType,
                  data: imageBuffer.toString('base64'),
                },
              },
              ...(lastMessage.trim() ? [{ text: lastMessage.trim() }] : []),
            ],
          },
        },
      ],
      localGpt: {
        provider: 'local-gpt',
        model: options.model,
        codexCommand: command.label,
        generatedImagePath: generatedImage.path,
      },
    }
  } finally {
    await rm(runDir, { recursive: true, force: true })
  }
}

function buildLocalGptPrompt(options: ImageCallOptions): string {
  const text = options.parts
    .map((part) => ('text' in part ? part.text.trim() : ''))
    .filter(Boolean)
    .join('\n\n')
  const sizeLine =
    options.outputWidth && options.outputHeight
      ? `The final image must be exactly ${options.outputWidth}x${options.outputHeight} pixels when the image generator supports exact sizing.`
      : ''

  return [
    text,
    sizeLine,
    'Use the local GPT image generator to create the final image. Save the generated image as an image file; do not create source code, HTML, or a mockup.',
  ]
    .filter(Boolean)
    .join('\n\n')
}

async function writeLocalGptInputImages(
  parts: readonly GeminiPart[],
  tempDir: string
): Promise<string[]> {
  const paths: string[] = []
  for (const part of parts) {
    if (!('inline_data' in part)) continue
    const inlineData = part.inline_data
    if (!inlineData.data.trim()) continue

    const extension = extensionForMimeType(inlineData.mime_type)
    const filePath = join(tempDir, `input-${paths.length}.${extension}`)
    await writeFile(filePath, Buffer.from(inlineData.data, 'base64'))
    paths.push(filePath)
  }
  return paths
}

type CodexCommand = {
  command: string
  argsPrefix: string[]
  label: string
}

type GeneratedImageSnapshot = {
  path: string
  mtimeMs: number
}

function resolveCodexCommand(): CodexCommand {
  const configured = process.env.LOCAL_CODEX_BIN?.trim()
  if (configured) {
    if (configured.endsWith('.js')) {
      return {
        command: process.execPath,
        argsPrefix: [configured],
        label: configured,
      }
    }
    return {
      command: configured,
      argsPrefix: [],
      label: configured,
    }
  }

  const desktopCodexExe = findDesktopCodexExe()
  if (desktopCodexExe) {
    return {
      command: desktopCodexExe,
      argsPrefix: [],
      label: desktopCodexExe,
    }
  }

  const cachedCodexJs = findCachedCodexJs()
  if (cachedCodexJs) {
    return {
      command: process.execPath,
      argsPrefix: [cachedCodexJs],
      label: cachedCodexJs,
    }
  }

  return {
    command: 'codex',
    argsPrefix: [],
    label: 'codex',
  }
}

function findDesktopCodexExe(): string | undefined {
  const localAppData = process.env.LOCALAPPDATA
  if (!localAppData) return undefined

  const binRoot = join(localAppData, 'OpenAI', 'Codex', 'bin')
  if (!existsSync(binRoot)) return undefined

  const candidates: GeneratedImageSnapshot[] = []
  const directExe = join(binRoot, 'codex.exe')
  if (existsSync(directExe)) {
    candidates.push({ path: directExe, mtimeMs: statSync(directExe).mtimeMs })
  }

  for (const child of readdirSync(binRoot, { withFileTypes: true })) {
    if (!child.isDirectory()) continue
    const candidate = join(binRoot, child.name, 'codex.exe')
    if (existsSync(candidate)) {
      candidates.push({
        path: candidate,
        mtimeMs: statSync(candidate).mtimeMs,
      })
    }
  }

  return candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0]?.path
}

function findCachedCodexJs(): string | undefined {
  const localAppData = process.env.LOCALAPPDATA
  if (!localAppData) return undefined

  const npxRoot = join(localAppData, 'npm-cache', '_npx')
  if (!existsSync(npxRoot)) return undefined

  const candidates: GeneratedImageSnapshot[] = []
  for (const child of readdirSync(npxRoot, { withFileTypes: true })) {
    if (!child.isDirectory()) continue
    const candidate = join(
      npxRoot,
      child.name,
      'node_modules',
      '@openai',
      'codex',
      'bin',
      'codex.js'
    )
    if (existsSync(candidate)) {
      candidates.push({
        path: candidate,
        mtimeMs: statSync(candidate).mtimeMs,
      })
    }
  }

  return candidates.sort((a, b) => b.mtimeMs - a.mtimeMs)[0]?.path
}

function snapshotGeneratedImages(): Set<string> {
  return new Set(readGeneratedImages().map((image) => image.path))
}

function findNewestGeneratedImage(
  before: Set<string>
): GeneratedImageSnapshot | undefined {
  return readGeneratedImages()
    .filter((image) => !before.has(image.path))
    .sort((a, b) => b.mtimeMs - a.mtimeMs)[0]
}

function readGeneratedImages(): GeneratedImageSnapshot[] {
  const root = join(
    process.env.CODEX_HOME ?? join(process.env.USERPROFILE ?? '', '.codex'),
    'generated_images'
  )
  if (!existsSync(root)) return []

  const results: GeneratedImageSnapshot[] = []
  const stack = [root]
  while (stack.length > 0) {
    const dir = stack.pop()
    if (!dir) continue

    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const fullPath = join(dir, entry.name)
      if (entry.isDirectory()) {
        stack.push(fullPath)
        continue
      }

      if (['.png', '.jpg', '.jpeg', '.webp'].includes(extname(entry.name).toLowerCase())) {
        results.push({
          path: fullPath,
          mtimeMs: statSync(fullPath).mtimeMs,
        })
      }
    }
  }

  return results
}

function runCommand(
  command: string,
  args: readonly string[],
  cwd: string
): Promise<{
  exitCode: number | null
  stdout: string
  stderr: string
}> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    })
    const timeoutMs = Number(
      process.env.LOCAL_CODEX_TIMEOUT_MS ?? 10 * 60 * 1000
    )
    const timeout = setTimeout(() => {
      child.kill('SIGKILL')
      reject(
        new LocalGptImageError(
          `Local GPT image generation timed out after ${timeoutMs}ms.`
        )
      )
    }, timeoutMs)
    let stdout = ''
    let stderr = ''
    child.stdout?.setEncoding('utf8')
    child.stderr?.setEncoding('utf8')
    child.stdout?.on('data', (chunk) => {
      stdout += chunk
    })
    child.stderr?.on('data', (chunk) => {
      stderr += chunk
    })
    child.on('error', (error) => {
      clearTimeout(timeout)
      reject(error)
    })
    child.on('close', (exitCode) => {
      clearTimeout(timeout)
      resolve({ exitCode, stdout, stderr })
    })
  })
}

function extensionForMimeType(mimeType: string): 'png' | 'jpg' | 'webp' {
  const normalized = mimeType.toLowerCase()
  if (normalized.includes('jpeg') || normalized.includes('jpg')) return 'jpg'
  if (normalized.includes('webp')) return 'webp'
  return 'png'
}

function mimeTypeForFile(filePath: string): string {
  const extension = extname(filePath).toLowerCase()
  if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg'
  if (extension === '.webp') return 'image/webp'
  return 'image/png'
}

export function extractGeminiImage(data: unknown): string | null {
  const candidates = (data as any)?.candidates
  if (!Array.isArray(candidates)) return null

  for (const candidate of candidates) {
    const parts = candidate?.content?.parts
    if (!Array.isArray(parts)) continue

    for (const part of parts) {
      const inlineData = part?.inlineData || part?.inline_data
      const imageData = inlineData?.data
      if (typeof imageData === 'string' && imageData.length > 0) {
        const mimeType =
          inlineData?.mimeType || inlineData?.mime_type || 'image/png'
        return `data:${mimeType};base64,${imageData}`
      }
    }
  }

  return null
}

export function extractGeminiText(data: unknown): string {
  const candidates = (data as any)?.candidates
  if (!Array.isArray(candidates)) return ''

  const chunks: string[] = []
  for (const candidate of candidates) {
    const parts = candidate?.content?.parts
    if (!Array.isArray(parts)) continue

    for (const part of parts) {
      if (typeof part?.text === 'string' && part.text.trim()) {
        chunks.push(part.text.trim())
      }
    }
  }

  return chunks.join('\n').trim()
}

export async function geminiErrorMessage(response: Response): Promise<string> {
  const fallback = `Gemini API request failed with status ${response.status}`
  try {
    const data = await response.json()
    return (
      data?.error?.message ||
      data?.message ||
      (typeof data === 'string' ? data : '') ||
      fallback
    )
  } catch {
    try {
      return (await response.text()) || fallback
    } catch {
      return fallback
    }
  }
}
