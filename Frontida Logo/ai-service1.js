// ══════════════════════════════════════════════════════════════════════
// ai-service.js — Frontida Pflegedokumentation
// Zentrales KI-Modul: OpenAI via Netlify Function, Prompts, Report-Generierung
// ══════════════════════════════════════════════════════════════════════
// Der OpenAI API-Key liegt NICHT im Frontend, sondern serverseitig in der
// Netlify Function unter netlify/functions/generate-report.js
// (env var OPENAI_API_KEY). Dieses Modul spricht ausschließlich den
// Netlify-Endpunkt an.
// ══════════════════════════════════════════════════════════════════════

// ── KONFIGURATION ────────────────────────────────────────────────────

const AI_ENDPOINT = '/.netlify/functions/generate-report';
const AI_RETRY_DELAYS_MS = [1200, 2500, 5000];


// ── PROMPTS (zentrale Definitionen) ──────────────────────────────────

const PFLEGE_REPORT_RULES =
  '- Ausschließlich auf Deutsch schreiben\n'
+ '- 4 bis 6 vollständige Sätze\n'
+ '- Keine Überschrift, kein Datum, keine Listen\n'
+ '- Immer "Patient/in" statt Namen verwenden\n'
+ '- Keine Satzfragmente, keine abgebrochenen Halbsätze, kein offenes Ende\n'
+ '- Der letzte Satz muss vollständig sein und mit einem Punkt enden\n'
+ '- Nur Informationen aus der Eingabe verwenden; keine erfundenen Vitalwerte oder Diagnosen\n'
+ '- Wenn Beschwerden, Sturz, Erbrechen, fehlende Nahrungsaufnahme oder verweigerte Medikamente erwähnt werden, diese klar und vollständig dokumentieren';

const SYSTEM_ROLLE = 'Du bist eine examinierte Pflegefachkraft bei einem deutschen ambulanten Pflegedienst.';

function buildCareReportPrompt(inputText, typ) {
  return SYSTEM_ROLLE + '\n'
    + 'Aufgabe: Erkenne die Sprache der Eingabe automatisch und schreibe daraus einen vollständigen professionellen Pflegebericht auf Deutsch.\n'
    + 'Berichtstyp: ' + typ + '\n'
    + 'Eingabe:\n"' + inputText + '"\n\n'
    + 'REGELN:\n'
    + PFLEGE_REPORT_RULES
    + '\n- Aus Stichpunkten oder Umgangssprache vollständige, professionelle Sätze bilden\n\n'
    + 'Output:';
}

function buildAudioReportPrompt(typ) {
  return SYSTEM_ROLLE + '\n'
    + 'Ein Mitarbeiter hat eine Sprachaufnahme in beliebiger Sprache aufgenommen.\n'
    + 'Aufgabe: Nutze das nachfolgende Transkript, verstehe den Inhalt und schreibe daraus einen vollständigen professionellen Pflegebericht auf Deutsch.\n'
    + 'Berichtstyp: ' + typ + '\n'
    + 'REGELN:\n'
    + PFLEGE_REPORT_RULES;
}

function buildAudioNotesPrompt() {
  return 'Gib den Inhalt der folgenden Sprachaufnahme als klare deutsche Notizen wieder.\n'
    + 'REGELN:\n'
    + '- Ausschließlich auf Deutsch schreiben\n'
    + '- Keine Überschrift\n'
    + '- Keine Erklärung über Sprache, Modell oder Audio\n'
    + '- Alle wichtigen Informationen behalten: Beschwerden, Sturz, Erbrechen, Essen, Trinken, Medikamente, Verhalten, Maßnahmen\n'
    + '- Nur den Inhalt wiedergeben, der in der Sprachaufnahme wirklich vorkommt\n'
    + '- Gib 1 bis 3 klare vollständige Sätze aus';
}

function buildRepairPrompt(cleaned, fallbackInput) {
  return 'Überarbeite den folgenden deutschen Pflegebericht zu einem vollständigen professionellen Pflegebericht.\n'
    + 'Wichtig:\n'
    + '- Keine neuen Fakten erfinden\n'
    + '- Alle Sätze vollständig formulieren\n'
    + '- 4 bis 6 vollständige Sätze\n'
    + '- Letzter Satz muss vollständig sein und mit Punkt enden\n'
    + '- Nur der fertige Berichtstext, keine Überschrift\n\n'
    + 'Notizen als Grundlage:\n"' + fallbackInput + '"\n\n'
    + 'Aktueller Bericht:\n"' + cleaned + '"\n\n'
    + 'Fertiger Bericht:';
}

function buildSuggestionPrompt(typ) {
  return 'Schreibe einen professionellen deutschen Pflegebericht (4-5 Sätze) für einen ambulanten Pflegedienst. '
    + 'Berichtstyp: ' + typ + '. '
    + 'Verwende "Patient/in" statt Namen. Nur der Berichtstext, keine Überschrift.';
}

function buildSISPrompt(berichtTexte) {
  return 'Du bist eine examinierte Pflegefachkraft. '
    + 'Erstelle eine vollständige, fachlich korrekte SIS (Strukturierte Informationssammlung) '
    + 'AUSSCHLIESSLICH auf Basis der folgenden Pflegeberichte. '
    + 'Wichtig: Verwende NUR Informationen aus diesen Berichten. Erfinde NICHTS. '
    + 'Wenn eine Information nicht in den Berichten vorhanden ist, schreibe: "Keine ausreichenden Angaben vorhanden."\n\n'
    + 'PFLEGEBERICHTE DES PATIENTEN:\n'
    + berichtTexte + '\n\n'
    + 'AUFGABE: Erstelle die SIS als gültiges JSON-Objekt (kein Markdown, keine Backticks, nur reines JSON):\n'
    + '{'
    + '"frage":"Was beschäftigt den Klienten laut Berichten? Wünsche, Bedürfnisse, Erwartungen.",'
    + '"tf1":"TF1 Kognition/Kommunikation: Ressourcen | Probleme/Risiken | Bedarf (nur aus Berichten)",'
    + '"tf2":"TF2 Mobilität/Beweglichkeit: Ressourcen | Probleme/Risiken | Bedarf",'
    + '"tf3":"TF3 Krankheit/Behandlung: Diagnosen, Medikamente, Therapien, Schmerzen aus Berichten",'
    + '"tf4":"TF4 Selbstversorgung: Körperpflege, Ernährung, Ausscheidung aus Berichten",'
    + '"tf5":"TF5 Soziale Beziehungen: Familie, Kontakte, Einsamkeit aus Berichten",'
    + '"tf6":"TF6 Haushalt: Reinigung, Einkauf, Versorgung aus Berichten",'
    + '"ressourcen":"Alle Stärken und Fähigkeiten des Klienten aus den Berichten",'
    + '"unterstuetzung":"Konkreter Unterstützungsbedarf aus den Berichten",'
    + '"massnahmen":"Pflegemaßnahmen die laut Berichten bereits durchgeführt werden oder geplant sind"'
    + '}';
}

const FALLBACK_SUGGESTION = 'Patient/in in gutem Allgemeinzustand angetroffen, kooperativ und freundlich. Pflegemaßnahmen planmäßig durchgeführt. Medikamente korrekt verabreicht. Vitalzeichen im Normbereich. Keine Auffälligkeiten festgestellt.';


// ── NETLIFY FUNCTION CLIENT ──────────────────────────────────────────

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isRetryableStatus(status) {
  return [429, 500, 502, 503, 504].includes(status);
}

// Low-level call to the Netlify function. Retries on transient errors.
async function callAIEndpoint(body, timeoutMs) {
  let lastMessage = 'KI nicht erreichbar.';

  for (let attempt = 0; attempt <= AI_RETRY_DELAYS_MS.length; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs || 60000);
    try {
      const r = await fetch(AI_ENDPOINT, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      clearTimeout(timeout);
      const data = await r.json().catch(() => ({}));
      if (r.ok) return data;
      lastMessage = data?.error || ('API Fehler ' + r.status);
      if (isRetryableStatus(r.status) && attempt < AI_RETRY_DELAYS_MS.length) {
        await sleep(AI_RETRY_DELAYS_MS[attempt]);
        continue;
      }
      break;
    } catch (e) {
      clearTimeout(timeout);
      lastMessage = e.name === 'AbortError'
        ? 'Zeitüberschreitung – bitte erneut versuchen'
        : (e.message || 'Netzwerkfehler');
      if (attempt < AI_RETRY_DELAYS_MS.length) {
        await sleep(AI_RETRY_DELAYS_MS[attempt]);
      }
    }
  }
  throw new Error(lastMessage);
}

// Haupt-Textgenerierung über die Netlify Function (OpenAI gpt-4o-mini)
async function callOpenAI(prompt, maxTokens) {
  const data = await callAIEndpoint({
    prompt: prompt,
    maxTokens: maxTokens || 400,
    temperature: 0.2
  }, 45000);
  return (data.text || '').trim();
}

// Rückwärtskompatibilität: callClaude / callGemini als Aliase
// Alle bestehenden Aufrufe im Haupt-Script funktionieren weiter
var callGemini = callOpenAI;
var callClaude = callOpenAI;


// ── REPORT-VERARBEITUNG ──────────────────────────────────────────────

function cleanGeneratedReport(text) {
  return String(text || '')
    .replace(/^#+\s*/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/^(Pflegebericht|Bericht|Report)\s*[:\-–]\s*/i, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function isUsefulAudioNotes(text) {
  const cleaned = cleanGeneratedReport(text);
  if (!cleaned) return false;
  if (cleaned.length < 20) return false;
  return !/^(sprachaufnahme|audio|mitarbeiter|routinebericht)\b/i.test(cleaned);
}

function isCompleteReport(text) {
  const cleaned = cleanGeneratedReport(text);
  if (!cleaned) return false;
  if (cleaned.length < 60) return false;
  if (!/[.!?…""]$/.test(cleaned)) return false;
  if ((cleaned.match(/[.!?](?=\s|$)/g) || []).length < 2) return false;
  return !/(und|oder|sowie|weil|dass|bei|mit|ohne|nach|vor|wegen|aufgrund|besteht|war|wurde|ist|hat|zeigt)\s*$/i.test(cleaned);
}

async function finalizeCareReport(text, fallbackInput, typ) {
  let cleaned = cleanGeneratedReport(text);
  if (isCompleteReport(cleaned)) return cleaned;

  // Repair-Versuch
  cleaned = cleanGeneratedReport(
    await callOpenAI(buildRepairPrompt(cleaned, fallbackInput), 500)
  );
  if (isCompleteReport(cleaned)) return cleaned;

  // Fallback auf Rohtext wenn kein Audio-Prefix
  if (fallbackInput && !/^Sprachaufnahme des Mitarbeiters zu\s+/i.test(cleanGeneratedReport(fallbackInput))) {
    return cleanGeneratedReport(fallbackInput);
  }
  return '';
}


// ── HIGH-LEVEL API ───────────────────────────────────────────────────

// Text-Eingabe → Pflegebericht
async function generateReport(inputText, typ) {
  const raw = await callOpenAI(buildCareReportPrompt(inputText, typ), 700);
  return finalizeCareReport(raw, inputText, typ);
}

// Audio (Base64) → Pflegebericht
// Die Netlify Function transkribiert das Audio (Whisper) und ruft
// anschließend gpt-4o-mini mit dem Transkript + Prompt auf.
async function generateReportFromAudio(base64, mimeType, typ) {
  // Schritt 1: Direkt Bericht aus Audio-Transkript
  const firstData = await callAIEndpoint({
    prompt: buildAudioReportPrompt(typ),
    maxTokens: 700,
    temperature: 0.2,
    audio: { base64: base64, mimeType: mimeType }
  }, 90000);
  let txt = (firstData.text || '').trim();

  // Schritt 2: Falls unvollständig → Notizen extrahieren → daraus Bericht
  if (!isCompleteReport(txt)) {
    const notesData = await callAIEndpoint({
      prompt: buildAudioNotesPrompt(),
      maxTokens: 500,
      temperature: 0.1,
      audio: { base64: base64, mimeType: mimeType }
    }, 90000);
    const notes = (notesData.text || '').trim();
    if (isUsefulAudioNotes(notes)) {
      txt = await callOpenAI(buildCareReportPrompt(notes, typ), 700);
    }
  }

  // Schritt 3: Finalisierung
  return finalizeCareReport(txt, '', typ);
}

// KI-Vorschlag für leeres Berichtsfeld
async function generateSuggestion(typ) {
  return callOpenAI(buildSuggestionPrompt(typ), 300);
}

// Berichte → SIS-JSON
async function generateSIS(berichte) {
  const berichtTexte = berichte.slice(0, 20).map(function(b, i) {
    return 'Bericht ' + (i + 1) + ' (' + b.dat + ', ' + b.autor + ', ' + b.typ + '): ' + b.text;
  }).join('\n\n');

  const raw = await callOpenAI(buildSISPrompt(berichtTexte), 2000);
  const clean = raw.replace(/```json|```/g, '').trim();
  return JSON.parse(clean);
}
