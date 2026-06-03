import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

async function loadHelper(env = {}) {
  const helperPath = path.resolve('app/api/_lib/gemini.ts')
  const source = await fs.readFile(helperPath, 'utf8')
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
    },
  })
  const module = { exports: {} }
  const context = {
    exports: module.exports,
    module,
    process: { env },
    Response,
    fetch: async () => {
      throw new Error('fetch should be mocked per test')
    },
  }

  vm.runInNewContext(outputText, context, { filename: helperPath })
  return module.exports
}

test('resolves a request Gemini API key before environment keys', async () => {
  const helper = await loadHelper({ GEMINI_API_KEY: 'env-gemini-key' })

  assert.equal(helper.resolveGeminiApiKey(' request-key '), 'request-key')
})

test('falls back to Gemini environment keys', async () => {
  const helper = await loadHelper({
    GEMINI_API_KEY: 'gemini-env-key',
    GOOGLE_API_KEY: 'google-env-key',
  })

  assert.equal(helper.resolveGeminiApiKey(undefined), 'gemini-env-key')
})

test('allows only Nano Banana image models and defaults unknown values', async () => {
  const helper = await loadHelper()

  assert.equal(helper.resolveGeminiModel('gemini-3-pro-image'), 'gemini-3-pro-image')
  assert.equal(helper.resolveGeminiModel('google/gemini-3-pro-image-preview'), 'gemini-3.1-flash-image')
  assert.equal(helper.resolveGeminiModel(undefined), 'gemini-3.1-flash-image')
})

test('converts image data URLs to Gemini inline data parts', async () => {
  const helper = await loadHelper()

  const part = JSON.parse(
    JSON.stringify(helper.dataUrlToGeminiPart('data:image/jpeg;base64,abc123'))
  )

  assert.deepEqual(part, {
    inline_data: {
      mime_type: 'image/jpeg',
      data: 'abc123',
    },
  })
})

test('extracts image and text from Gemini generateContent responses', async () => {
  const helper = await loadHelper()
  const response = {
    candidates: [
      {
        content: {
          parts: [
            { text: 'first line' },
            { inlineData: { mimeType: 'image/png', data: 'abc123' } },
            { text: 'second line' },
          ],
        },
      },
    ],
  }

  assert.equal(helper.extractGeminiImage(response), 'data:image/png;base64,abc123')
  assert.equal(helper.extractGeminiText(response), 'first line\nsecond line')
})

test('image generation config omits unsupported REST image response fields', async () => {
  const helper = await loadHelper()
  const config = JSON.parse(
    JSON.stringify(helper.imageGenerationConfig('16:9', 'gemini-3.1-flash-image'))
  )

  assert.equal(Object.hasOwn(config, 'responseModalities'), false)
  assert.equal(Object.hasOwn(config, 'responseFormat'), false)
  assert.deepEqual(config, {})
})
