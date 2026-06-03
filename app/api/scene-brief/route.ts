import { NextRequest, NextResponse } from 'next/server'
import {
  callGeminiGenerateContent,
  extractGeminiText,
  GeminiApiError,
  resolveGeminiApiKey,
  resolveGeminiModel,
} from '@/app/api/_lib/gemini'

const artStyleDescriptions: Record<string, string> = {
  cinematic: 'cinematic photography with dramatic lighting and film grain',
  vintage: 'vintage film photography with faded colors and retro feel',
  'black-white': 'black and white photography with rich contrast',
  'oil-painting': 'oil painting style with visible brush strokes and rich textures',
  watercolor: 'watercolor painting with soft washes and flowing colors',
  impressionism: 'impressionist painting style with loose brushwork',
  'digital-art': 'digital art with smooth gradients and modern aesthetics',
  cyberpunk: 'cyberpunk style with neon colors and futuristic elements',
  vaporwave: 'vaporwave aesthetic with pastel colors and retro-futuristic vibes',
  'low-poly': 'low poly 3D art with geometric faceted surfaces',
  'pixel-art': 'pixel art style with retro video game aesthetics',
  '3d-render': '3D rendered look with realistic lighting and materials',
  anime: 'anime/manga style with bold lines and vibrant colors',
  cartoon: 'cartoon illustration with exaggerated features',
  'studio-ghibli': 'Studio Ghibli animation style with whimsical hand-drawn aesthetics',
  fantasy: 'fantasy art with magical and ethereal elements',
  'sci-fi': 'science fiction with futuristic technology and environments',
}

export async function POST(request: NextRequest) {
  try {
    const { anchorPrompt, artStyle, apiKey, model } = await request.json()

    if (!anchorPrompt || typeof anchorPrompt !== 'string' || !anchorPrompt.trim()) {
      return NextResponse.json(
        { error: 'Missing anchor prompt' },
        { status: 400 }
      )
    }

    const geminiKey = resolveGeminiApiKey(apiKey)

    if (!geminiKey) {
      return NextResponse.json(
        { error: 'Gemini API key missing. Add one in Settings.' },
        { status: 401 }
      )
    }

    const modelId = resolveGeminiModel(model)

    const styleLine =
      artStyle && artStyleDescriptions[artStyle]
        ? `\nArt style: ${artStyleDescriptions[artStyle]}.`
        : ''

    const systemPrompt = `You help game designers build multi-layer parallax backgrounds. Given the prompt used for the NEAR (foreground) anchor layer, write a concise SCENE BRIEF that every other layer (mid-ground, far distance, sky/back) must follow so the final composite feels like one cohesive world.

Rules for your brief:
- 3–5 sentences, plain text only — no markdown, no bullet lists, no headers.
- Capture: setting/environment, time of day, lighting quality, color palette (name specific colors), art style, mood/atmosphere.
- Lighting must be ambient and horizontally even (no sun/moon on one side) because the sky layer will tile horizontally in-game.
- Write as instructions an artist would follow when painting matching layers behind the foreground — not layer-specific composition rules.
- Do NOT repeat the anchor prompt verbatim; distill the shared art direction.`

    const userPrompt = `Near (foreground) layer prompt:
"${anchorPrompt.trim()}"${styleLine}

Write the shared scene brief for all parallax layers.`

    let data: unknown
    try {
      data = await callGeminiGenerateContent({
        apiKey: geminiKey,
        model: modelId,
        parts: [
          { text: `System instructions:\n${systemPrompt}` },
          { text: userPrompt },
        ],
        generationConfig: {
          maxOutputTokens: 400,
          temperature: 0.4,
        },
      })
    } catch (error) {
      if (error instanceof GeminiApiError) {
        return NextResponse.json(
          { error: error.message || 'Failed to generate scene brief' },
          { status: error.status }
        )
      }
      throw error
    }

    const sceneBrief = extractGeminiText(data)
    if (!sceneBrief) {
      return NextResponse.json(
        { error: 'No scene brief returned from model' },
        { status: 500 }
      )
    }

    return NextResponse.json({ sceneBrief })
  } catch (error) {
    console.error('Error in scene-brief route:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
