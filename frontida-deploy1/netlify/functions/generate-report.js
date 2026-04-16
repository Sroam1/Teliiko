// ══════════════════════════════════════════════════════════════════════
// netlify/functions/generate-report.js
// Backend proxy for OpenAI — keeps API key off the frontend.
// ══════════════════════════════════════════════════════════════════════
//
// Expected request body (JSON):
//   {
//     prompt:       string,            // required for text generation
//     maxTokens?:   number,            // default 700
//     temperature?: number,            // default 0.2
//     systemPrompt?: string,           // optional override for default system prompt
//     audio?: { base64, mimeType }     // optional — will be transcribed first
//                                      // and the transcript appended to the prompt
//     image?: { dataUrl }              // optional — vision analysis via gpt-4o-mini
//                                      // dataUrl = "data:image/jpeg;base64,...."
//     transcribeOnly?: boolean         // optional — if true AND audio is given,
//                                      // skip chat completion and return only
//                                      // the raw Whisper transcript in `text`.
//   }
//
// Response: { text: string }
//
// Required Netlify env var: OPENAI_API_KEY
// ══════════════════════════════════════════════════════════════════════

const OPENAI_CHAT_URL = 'https://api.openai.com/v1/chat/completions';
const OPENAI_TRANSCRIBE_URL = 'https://api.openai.com/v1/audio/transcriptions';
const CHAT_MODEL = "gpt-3.5-turbo";
const TRANSCRIBE_MODEL = 'whisper-1';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS'
};

function json(statusCode, body) {
  return {
    statusCode,
    headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
    body: JSON.stringify(body)
  };
}

function extFromMime(mime) {
  if (!mime) return 'webm';
  if (mime.includes('webm')) return 'webm';
  if (mime.includes('ogg'))  return 'ogg';
  if (mime.includes('wav'))  return 'wav';
  if (mime.includes('mp4'))  return 'mp4';
  if (mime.includes('m4a'))  return 'm4a';
  if (mime.includes('mpeg')) return 'mp3';
  if (mime.includes('mp3'))  return 'mp3';
  return 'webm';
}

async function transcribeAudio(base64, mimeType, apiKey) {
  const buffer = Buffer.from(base64, 'base64');
  const ext = extFromMime(mimeType);
  const filename = 'audio.' + ext;

  // Build multipart/form-data manually — Netlify's Node runtime has Blob/FormData
  // available on recent versions.
  const form = new FormData();
  const blob = new Blob([buffer], { type: mimeType || 'audio/webm' });
  form.append('file', blob, filename);
  form.append('model', TRANSCRIBE_MODEL);

  const r = await fetch(OPENAI_TRANSCRIBE_URL, {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + apiKey },
    body: form
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = data?.error?.message || ('Transcription failed (' + r.status + ')');
    throw new Error(msg);
  }
  return (data.text || '').trim();
}

const DEFAULT_SYSTEM_PROMPT = `Du schreibst einfache und kurze Pflegeberichte auf Deutsch.

Der Input kann auf Griechisch, Deutsch oder gemischt sein.

AUFGABE:
- Verstehe den Text auch wenn er gemischt ist (Griechisch + deutsche Stichwörter)
- Erkenne typische Pflegebegriffe wie Körperpflege, Medikation, Mobilisation, Hautzustand, Essen, Stimmung, auch wenn sie falsch geschrieben sind
- Formuliere alles als einfachen und alltagstauglichen Pflegebericht auf Deutsch

WICHTIG:
- Nur das schreiben, was gesagt wurde
- Nichts erfinden
- Keine unnötigen Ergänzungen
- Keine langen Erklärungen
- Kein Arztstil
- Einfach und direkt schreiben
- Maximal 2 bis 4 kurze Sätze
- Kleine sinnvolle Ergänzungen sind nur erlaubt, wenn sie wirklich naheliegend sind

Wenn Informationen fehlen, lasse sie weg.

Beispiele:

Input: "δεν ηθελε Kleinekörperpflege, Medigabe εγινε"
Output: "Körperpflege abgelehnt. Medikamente verabreicht."

Input: "mobilisation εγινε, ποναγε λιγο"
Output: "Mobilisation durchgeführt. Patient klagt über leichte Schmerzen."

Input: "πήρε τα χαπια και δεν ηθελε μπανιο"
Output: "Patient hat Medikamente eingenommen. Körperpflege abgelehnt."`;

async function chatCompletion(prompt, maxTokens, temperature, apiKey, systemPrompt, imageDataUrl) {
  const sysContent = (typeof systemPrompt === 'string' && systemPrompt.trim())
    ? systemPrompt
    : DEFAULT_SYSTEM_PROMPT;

  // User message: plain text when no image, multimodal content array otherwise.
  const userContent = imageDataUrl
    ? [
        { type: 'text', text: prompt || '' },
        { type: 'image_url', image_url: { url: imageDataUrl } }
      ]
    : prompt;

  const r = await fetch(OPENAI_CHAT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + apiKey
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages: [
        { role: 'system', content: sysContent },
        { role: 'user',   content: userContent }
      ],
      max_tokens: maxTokens,
      temperature: temperature
    })
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = data?.error?.message || ('OpenAI error (' + r.status + ')');
    throw new Error(msg);
  }
  const text = data?.choices?.[0]?.message?.content || '';
  return text.trim();
}

exports.handler = async function (event) {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204, headers: CORS_HEADERS, body: '' };
  }
  if (event.httpMethod !== 'POST') {
    return json(405, { error: 'Method not allowed' });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return json(500, { error: 'Server misconfigured: OPENAI_API_KEY missing.' });
  }

  let payload;
  try {
    payload = JSON.parse(event.body || '{}');
  } catch (_e) {
    return json(400, { error: 'Invalid JSON body.' });
  }

  const prompt         = typeof payload.prompt === 'string' ? payload.prompt : '';
  const maxTokens      = Number.isFinite(payload.maxTokens) ? payload.maxTokens : 700;
  const temperature    = Number.isFinite(payload.temperature) ? payload.temperature : 0.2;
  const systemPrompt   = typeof payload.systemPrompt === 'string' ? payload.systemPrompt : '';
  const audio          = payload.audio;
  const image          = payload.image;
  const transcribeOnly = payload.transcribeOnly === true;

  if (!prompt && !audio && !image) {
    return json(400, { error: 'prompt, audio or image required.' });
  }

  try {
    // Kurzschluss: nur Whisper-Transkript zurückgeben (für Voice-Input-Felder)
    if (transcribeOnly) {
      if (!audio || !audio.base64) {
        return json(400, { error: 'audio required for transcribeOnly.' });
      }
      const transcript = await transcribeAudio(audio.base64, audio.mimeType, apiKey);
      return json(200, { text: transcript });
    }

    let finalPrompt = prompt;

    if (audio && audio.base64) {
      const transcript = await transcribeAudio(audio.base64, audio.mimeType, apiKey);
      finalPrompt = (prompt ? prompt + '\n\n' : '')
        + 'Transkript der Sprachaufnahme:\n"' + transcript + '"';
    }

    const imageDataUrl = (image && typeof image.dataUrl === 'string') ? image.dataUrl : '';
    const text = await chatCompletion(finalPrompt, maxTokens, temperature, apiKey, systemPrompt, imageDataUrl);
    return json(200, { text });
  } catch (e) {
    return json(502, { error: e.message || 'Upstream error' });
  }
};
