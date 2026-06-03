export const DEFAULT_GEMINI_IMAGE_MODEL = 'gemini-3.1-flash-image'

export const GEMINI_IMAGE_MODELS = [
  'gemini-3-pro-image',
  'gemini-3.1-flash-image',
  'gemini-2.5-flash-image',
] as const

export type GeminiImageModel = (typeof GEMINI_IMAGE_MODELS)[number]

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
  responseFormat?: {
    image?: {
      aspectRatio?: string
      imageSize?: string
    }
  }
}

export type GeminiCallOptions = {
  apiKey?: unknown
  model?: unknown
  parts: GeminiPart[]
  generationConfig?: GeminiGenerationConfig
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
  const image: NonNullable<
    NonNullable<GeminiGenerationConfig['responseFormat']>['image']
  > = {}

  if (aspectRatio) image.aspectRatio = aspectRatio
  if (model === 'gemini-3-pro-image' || model === 'gemini-3.1-flash-image') {
    image.imageSize = '1K'
  }

  return {
    ...(Object.keys(image).length
      ? {
          responseFormat: {
            image,
          },
        }
      : {}),
  }
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
