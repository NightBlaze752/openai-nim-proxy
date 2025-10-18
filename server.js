// server.js — OpenAI -> NVIDIA NIM proxy (flex)
// - Display reasoning for models that return it (wraps in <think>...</think>)
// - Optional vendor hints via env to try to unlock reasoning on specific models
// - Clean error handling + streaming fix + model aliases
//
// Env vars (Render -> Environment):
//   NIM_API_KEY                 required
//   SHOW_REASONING_MODELS       e.g. "deepseek,terminus,r1"
//   MODEL_MAP_OVERRIDES         JSON: {"gpt-4o":"deepseek-ai/deepseek-v3.1", ...}
//   // Optional reasoning attempts (top-level merge into request):
//   REQUEST_MERGE_BY_MODEL      JSON: {"deepseek-ai/deepseek-v3.1":{"reasoning":{"effort":"medium"},"enable_reasoning":true,"include_reasoning":true,"chat_template_kwargs":{"thinking":true}},"deepseek-ai/deepseek-v3.1-terminus":{...}}
//   REQUEST_MERGE_GLOBAL        JSON (merged for all models)
//   // Optional: put extra vendor params under extra_body for providers that expect it there:
//   EXTRA_BODY_BY_MODEL         JSON: {"deepseek-ai/deepseek-v3.1":{"chat_template_kwargs":{"thinking":true}}}
//   EXTRA_BODY_GLOBAL           JSON
//   THINK_OPEN_TAG              default "<think>"
//   THINK_CLOSE_TAG             default "</think>"
//   NIM_API_BASE                default "https://integrate.api.nvidia.com/v1"
//   ENABLE_THINKING_MODE        "true" or "false" (default false) adds extra_body.chat_template_kwargs.thinking=true

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY || '';

const ENABLE_THINKING_MODE = String(process.env.ENABLE_THINKING_MODE || 'false').toLowerCase() === 'true';

const SHOW_REASONING_MODELS = (process.env.SHOW_REASONING_MODELS || '')
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

const THINK_OPEN = process.env.THINK_OPEN_TAG || '<think>';
const THINK_CLOSE = process.env.THINK_CLOSE_TAG || '</think>';

function parseJSONEnv(name) {
  if (!process.env[name]) return null;
  try {
    return JSON.parse(process.env[name]);
  } catch {
    console.warn(`Invalid JSON in ${name}. Ignoring.`);
    return null;
  }
}

const REQUEST_MERGE_GLOBAL = parseJSONEnv('REQUEST_MERGE_GLOBAL') || {};
const REQUEST_MERGE_BY_MODEL = parseJSONEnv('REQUEST_MERGE_BY_MODEL') || {};
const EXTRA_BODY_GLOBAL = parseJSONEnv('EXTRA_BODY_GLOBAL') || {};
const EXTRA_BODY_BY_MODEL = parseJSONEnv('EXTRA_BODY_BY_MODEL') || {};

function deepMerge(target, source) {
  if (!source || typeof source !== 'object') return target;
  for (const key of Object.keys(source)) {
    const srcVal = source[key];
    const tgtVal = target[key];
    if (srcVal && typeof srcVal === 'object' && !Array.isArray(srcVal)) {
      target[key] = deepMerge(tgtVal && typeof tgtVal === 'object' ? tgtVal : {}, srcVal);
    } else {
      target[key] = srcVal;
    }
  }
  return target;
}

function shouldShowReasoning(nimModelId) {
  if (!nimModelId || SHOW_REASONING_MODELS.length === 0) return false;
  const id = String(nimModelId).toLowerCase();
  return SHOW_REASONING_MODELS.some(token => id.includes(token));
}

// Defaults — override with MODEL_MAP_OVERRIDES
const DEFAULT_MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o-mini': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4': 'deepseek-ai/deepseek-r1-0528',
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct'
};

let MODEL_MAPPING = { ...DEFAULT_MODEL_MAPPING };
const MODEL_MAP_OVERRIDES = parseJSONEnv('MODEL_MAP_OVERRIDES');
if (MODEL_MAP_OVERRIDES) {
  MODEL_MAPPING = { ...MODEL_MAPPING, ...MODEL_MAP_OVERRIDES };
  console.log('Loaded MODEL_MAP_OVERRIDES:', MODEL_MAPPING);
}

// For robustness: pick up alternative reasoning field names if provider uses them
const REASONING_FIELDS = ['reasoning_content', 'reasoning', 'thoughts', 'thinking', 'chain_of_thought'];

function extractReasoningFromDelta(delta) {
  let buf = '';
  for (const f of REASONING_FIELDS) {
    if (typeof delta[f] === 'string' && delta[f].length) {
      buf += delta[f];
      delete delta[f];
    }
  }
  return buf;
}

function extractReasoningFromMessage(msg) {
  if (!msg || typeof msg !== 'object') return '';
  for (const f of REASONING_FIELDS) {
    const v = msg[f];
    if (typeof v === 'string' && v.length) return v;
  }
  return '';
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI->NIM Proxy',
    thinking_mode: ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED',
    show_reasoning_allowlist: SHOW_REASONING_MODELS,
    has_nim_key: !!NIM_API_KEY,
    merges: {
      request_global: Object.keys(REQUEST_MERGE_GLOBAL).length,
      request_by_model: Object.keys(REQUEST_MERGE_BY_MODEL).length,
      extra_body_global: Object.keys(EXTRA_BODY_GLOBAL).length,
      extra_body_by_model: Object.keys(EXTRA_BODY_BY_MODEL).length
    }
  });
});

app.get('/v1/models', async (req, res) => {
  try {
    const r = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` }
    });

    const upstream = r.data?.data || r.data || [];
    const list = Array.isArray(upstream) ? upstream : upstream.data || [];

    const aliases = Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'openai-nim-proxy-alias'
    }));

    res.json({ object: 'list', data: [...aliases, ...list] });
  } catch (error) {
    const status = error.response?.status || 500;
    const raw = error.response?.data;
    const msg =
      (typeof raw === 'string' && raw) ||
      raw?.error?.message ||
      raw?.message ||
      error.message ||
      'models error';
    console.error('Models passthrough error:', { status, message: msg });
    res.status(status).json({ error: { message: msg, type: 'models_error', code: status } });
  }
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body || {};
    if (!model || !Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: 'Missing required fields: model, messages[]', type: 'invalid_request_error', code: 400 }
      });
    }

    const nimModel = MODEL_MAPPING[model] || model;

    let nimRequest = {
      model: nimModel,
      messages,
      temperature: typeof temperature === 'number' ? temperature : 0.6,
      max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
      stream: !!stream
    };

    // Optional thinking mode (generic hint — may be ignored by many models)
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = nimRequest.extra_body || {};
      nimRequest.extra_body.chat_template_kwargs = nimRequest.extra_body.chat_template_kwargs || {};
      nimRequest.extra_body.chat_template_kwargs.thinking = true;
    }

    // Apply global and per-model top-level merges (experimental vendor hints)
    if (REQUEST_MERGE_GLOBAL && Object.keys(REQUEST_MERGE_GLOBAL).length) {
      nimRequest = deepMerge(nimRequest, JSON.parse(JSON.stringify(REQUEST_MERGE_GLOBAL)));
    }
    const reqMergeForModel = REQUEST_MERGE_BY_MODEL[nimModel];
    if (reqMergeForModel) {
      nimRequest = deepMerge(nimRequest, JSON.parse(JSON.stringify(reqMergeForModel)));
    }

    // Apply extra_body merges (for providers that expect options there)
    if (EXTRA_BODY_GLOBAL && Object.keys(EXTRA_BODY_GLOBAL).length) {
      nimRequest.extra_body = deepMerge(nimRequest.extra_body || {}, JSON.parse(JSON.stringify(EXTRA_BODY_GLOBAL)));
    }
    const extraForModel = EXTRA_BODY_BY_MODEL[nimModel];
    if (extraForModel) {
      nimRequest.extra_body = deepMerge(nimRequest.extra_body || {}, JSON.parse(JSON.stringify(extraForModel)));
    }

    const axiosConfig = {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      validateStatus: s => s < 500
    };

    const upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, axiosConfig);

    if (upstream.status >= 400) {
      return res.status(upstream.status).json(upstream.data);
    }

    const showReasoning = shouldShowReasoning(nimModel);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      let buffer = '';
      let reasoningBuf = '';
      let emittedReasoningBlock = false;

      function emit(obj) {
        res.write(`data: ${JSON.stringify(obj)}\n\n`);
      }
      function emitReasoningBlockIfNeeded() {
        if (!showReasoning || !reasoningBuf || emittedReasoningBlock) return;
        const block = `${THINK_OPEN}\n${reasoningBuf}\n${THINK_CLOSE}\n\n`;
        const synthetic = {
          id: `chunk-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{ index: 0, delta: { content: block }, finish_reason: null }]
        };
        emit(synthetic);
        emittedReasoningBlock = true;
      }

      upstream.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();

          if (payload === '[DONE]') {
            emitReasoningBlockIfNeeded();
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(payload);
            const delta = data?.choices?.[0]?.delta || {};

            if (showReasoning) {
              const r = extractReasoningFromDelta(delta);
              if (r) {
                reasoningBuf += r;
                // if this chunk was only reasoning, hold it until we have content
                const onlyReasoning = !delta.content || delta.content.length === 0;
                if (onlyReasoning) continue;
              }
              if (delta.content && !emittedReasoningBlock && reasoningBuf.length) {
                emitReasoningBlockIfNeeded();
              }
            }

            // always hide any leftover reasoning fields
            for (const f of REASONING_FIELDS) if (f in delta) delete delta[f];

            emit(data);
          } catch {
            res.write(line + '\n');
          }
        }
      });

      upstream.data.on('end', () => res.end());
      upstream.data.on('error', (err) => {
        console.error('Stream error:', err?.message || err);
        res.end();
      });
      return;
    }

    // Non-streaming
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: (upstream.data?.choices || []).map((choice, idx) => {
        const role = choice?.message?.role || 'assistant';
        let content = choice?.message?.content || '';

        if (showReasoning) {
          const r = extractReasoningFromMessage(choice?.message);
          if (r) content = `${THINK_OPEN}\n${r}\n${THINK_CLOSE}\n\n${content}`;
        }

        return {
          index: choice?.index ?? idx,
          message: { role, content },
          finish_reason: choice?.finish_reason || 'stop'
        };
      }),
      usage: upstream.data?.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    res.json(openaiResponse);
  } catch (error) {
    const status = error.response?.status || 500;
    const raw = error.response?.data;
    const safeMsg =
      (typeof raw === 'string' && raw) ||
      (raw && typeof raw.error?.message === 'string' && raw.error.message) ||
      (raw && typeof raw.message === 'string' && raw.message) ||
      (typeof error.message === 'string' && error.message) ||
      'Internal server error';

    console.error('Proxy error:', { status, message: safeMsg });
    res.status(status).json({ error: { message: safeMsg, type: 'invalid_request_error', code: status } });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI->NIM Proxy running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Reasoning allowlist: ${SHOW_REASONING_MODELS.length ? SHOW_REASONING_MODELS.join(', ') : 'OFF'}`);
});
