const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '4mb' }));

// -------------------- ENV --------------------
const NIM_API_BASE = String(process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1')
  .trim()
  .replace(/\/+$/, '');

const NIM_API_KEY = process.env.NIM_API_KEY || '';

const ENABLE_THINKING_MODE =
  process.env.ENABLE_THINKING_MODE == null
    ? true
    : String(process.env.ENABLE_THINKING_MODE).toLowerCase() === 'true';

const DEBUG_REASONING = String(process.env.DEBUG_REASONING || 'false').toLowerCase() === 'true';

// Example: "deepseek-v3.1,deepseek-v3.2,terminus,r1,glm"
const SHOW_REASONING_MODELS = (process.env.SHOW_REASONING_MODELS || 'deepseek-v3.1,deepseek-v3.2,terminus,r1,glm')
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

// If true, GLM reasoning is streamed as normal content (very visible).
// If false, reasoning is buffered and emitted as one <think> block.
const GLM_REASONING_AS_CONTENT = String(process.env.GLM_REASONING_AS_CONTENT || 'false').toLowerCase() === 'true';

// NVIDIA sample uses clear_thinking:false. You can flip to true if you see looping.
const GLM_CLEAR_THINKING =
  process.env.GLM_CLEAR_THINKING == null
    ? false
    : String(process.env.GLM_CLEAR_THINKING).toLowerCase() === 'true';

const THINK_OPEN = process.env.THINK_OPEN_TAG || '<think>';
const THINK_CLOSE = process.env.THINK_CLOSE_TAG || '</think>';

// Upstream model ids
const GLM_UPSTREAM_ID = 'z-ai/glm-5.1';

// -------------------- HELPERS --------------------
function parseJSONEnv(name) {
  if (!process.env[name]) return null;
  try {
    return JSON.parse(process.env[name]);
  } catch {
    console.warn(`Invalid JSON in ${name}. Ignoring.`);
    return null;
  }
}

function cloneJSON(x) {
  return x == null ? x : JSON.parse(JSON.stringify(x));
}

function deepMerge(target, source) {
  const out = target && typeof target === 'object' ? target : {};
  if (!source || typeof source !== 'object') return out;

  for (const key of Object.keys(source)) {
    const srcVal = source[key];
    const tgtVal = out[key];
    if (srcVal && typeof srcVal === 'object' && !Array.isArray(srcVal)) {
      out[key] = deepMerge(tgtVal && typeof tgtVal === 'object' ? tgtVal : {}, srcVal);
    } else {
      out[key] = srcVal;
    }
  }
  return out;
}

function isGlm(nimModelId) {
  return String(nimModelId || '').toLowerCase().startsWith('z-ai/glm');
}

function normalizeMappedModelId(modelId) {
  const id = String(modelId || '').trim().toLowerCase();

  // Map older GLM aliases to GLM-5.1
  if (
    id === 'z-ai/glm5' || id === 'z-ai/glm4.7' ||
    id === 'glm5' || id === 'glm-5' || id === 'glm5.0' || id === 'glm-5.0' ||
    id === 'glm4.7' || id === 'glm-4.7'
  ) return GLM_UPSTREAM_ID;

  // GLM-5.1 direct
  if (id === 'z-ai/glm-5.1' || id === 'glm5.1' || id === 'glm-5.1') return GLM_UPSTREAM_ID;

  return modelId;
}

function shouldShowReasoning(nimModelId) {
  if (!nimModelId || SHOW_REASONING_MODELS.length === 0) return false;
  const id = String(nimModelId).toLowerCase();
  return SHOW_REASONING_MODELS.some(token => id.includes(token));
}

function reasoningAsContentModel(nimModelId) {
  return isGlm(nimModelId) && GLM_REASONING_AS_CONTENT;
}

const REASONING_FIELDS = [
  'reasoning_content',
  'reasoning',
  'reasoning_text',
  'thinking',
  'thoughts',
  'chain_of_thought',
  'analysis'
];

function textFromAny(v) {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return v.map(textFromAny).filter(Boolean).join('');
  if (typeof v === 'object') {
    if (typeof v.text === 'string') return v.text;
    if (typeof v.content === 'string') return v.content;
    if (Array.isArray(v.content)) return v.content.map(textFromAny).filter(Boolean).join('');
    return Object.values(v).map(textFromAny).filter(Boolean).join('');
  }
  return '';
}

function extractReasoningFromDelta(delta) {
  if (!delta || typeof delta !== 'object') return '';
  let buf = '';
  for (const f of REASONING_FIELDS) {
    if (f in delta) {
      const extracted = textFromAny(delta[f]);
      if (extracted) buf += extracted;
      delete delta[f];
    }
  }
  return buf;
}

function extractReasoningFromMessage(msg) {
  if (!msg || typeof msg !== 'object') return '';
  let buf = '';
  for (const f of REASONING_FIELDS) {
    if (f in msg) {
      const extracted = textFromAny(msg[f]);
      if (extracted) buf += extracted;
    }
  }
  return buf;
}

// ----- STREAM-SAFE upstream error forwarding -----
function isReadableStream(x) {
  return x && typeof x === 'object' && typeof x.on === 'function' && typeof x.pipe === 'function';
}

function streamToString(stream, limitBytes = 2 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let total = 0;
    let out = '';
    stream.on('data', (chunk) => {
      const s = chunk.toString('utf8');
      total += Buffer.byteLength(s, 'utf8');
      if (total > limitBytes) {
        reject(new Error(`Upstream error body exceeded ${limitBytes} bytes`));
        stream.destroy();
        return;
      }
      out += s;
    });
    stream.on('end', () => resolve(out));
    stream.on('error', reject);
  });
}

async function sendUpstreamError(res, status, data, fallbackMessage = 'Upstream error') {
  try {
    if (isReadableStream(data)) {
      const text = await streamToString(data);
      try {
        return res.status(status).json(JSON.parse(text));
      } catch {
        return res.status(status).json({ error: { message: text || fallbackMessage, type: 'upstream_error', code: status } });
      }
    }

    if (typeof data === 'string') {
      try {
        return res.status(status).json(JSON.parse(data));
      } catch {
        return res.status(status).json({ error: { message: data || fallbackMessage, type: 'upstream_error', code: status } });
      }
    }

    if (data && typeof data === 'object') return res.status(status).json(data);

    return res.status(status).json({ error: { message: fallbackMessage, type: 'upstream_error', code: status } });
  } catch (e) {
    return res.status(500).json({ error: { message: e?.message || 'Failed to forward upstream error', type: 'proxy_error', code: 500 } });
  }
}

// -------------------- MODEL MAPPING --------------------
const DEFAULT_MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o-mini': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4': 'deepseek-ai/deepseek-r1-0528',
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',

  'deepseek-v3.1': 'deepseek-ai/deepseek-v3.1',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',
  'deepseek-v3.1-terminus': 'deepseek-ai/deepseek-v3.1-terminus',
  'deepseek-r1': 'deepseek-ai/deepseek-r1-0528',

  // v4 pro alias
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',

  // GLM aliases -> 5.1
  'glm5.1': GLM_UPSTREAM_ID,
  'glm-5.1': GLM_UPSTREAM_ID,
  'glm5': GLM_UPSTREAM_ID,
  'glm-5': GLM_UPSTREAM_ID,
  'glm5.0': GLM_UPSTREAM_ID,
  'glm-5.0': GLM_UPSTREAM_ID,
  'glm4.7': GLM_UPSTREAM_ID,
  'glm-4.7': GLM_UPSTREAM_ID,

  [GLM_UPSTREAM_ID]: GLM_UPSTREAM_ID
};

let MODEL_MAPPING = { ...DEFAULT_MODEL_MAPPING };
const MODEL_MAP_OVERRIDES = parseJSONEnv('MODEL_MAP_OVERRIDES');
if (MODEL_MAP_OVERRIDES && typeof MODEL_MAP_OVERRIDES === 'object') {
  MODEL_MAPPING = { ...MODEL_MAPPING, ...MODEL_MAP_OVERRIDES };
}

// -------------------- MERGE CONFIG FROM ENV --------------------
const REQUEST_MERGE_GLOBAL = parseJSONEnv('REQUEST_MERGE_GLOBAL') || { top_k: -1 };
const REQUEST_MERGE_BY_MODEL = parseJSONEnv('REQUEST_MERGE_BY_MODEL') || {};
const EXTRA_BODY_GLOBAL = parseJSONEnv('EXTRA_BODY_GLOBAL') || {};
const EXTRA_BODY_BY_MODEL = parseJSONEnv('EXTRA_BODY_BY_MODEL') || {};

function getPerModelConfig(map, nimModel) {
  if (!nimModel) return null;
  if (map[nimModel]) return map[nimModel];
  if (nimModel === 'deepseek-ai/deepseek-v3.2' && map['deepseek-ai/deepseek-v3.1']) return map['deepseek-ai/deepseek-v3.1'];
  return null;
}

function stripUnsupportedGlmTopLevelParams(nimRequest) {
  // NVIDIA now rejects these top-level keys for GLM:
  delete nimRequest.enable_reasoning;
  delete nimRequest.enable_thinking;

  // These are “maybe supported” but safest to remove for GLM if you’re getting validation errors:
  // If you later confirm they are accepted, you can keep them.
  delete nimRequest.include_reasoning;
  delete nimRequest.reasoning;
}

// -------------------- ROUTES --------------------
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI->NIM Proxy',
    thinking_mode: ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED',
    show_reasoning_allowlist: SHOW_REASONING_MODELS,
    glm_target: GLM_UPSTREAM_ID,
    glm_clear_thinking: GLM_CLEAR_THINKING,
    glm_reasoning_as_content: GLM_REASONING_AS_CONTENT,
    debug_reasoning: DEBUG_REASONING,
    nim_api_base: NIM_API_BASE,
    has_nim_key: !!NIM_API_KEY
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const body = req.body || {};
    const { model, messages, stream } = body;

    if (!model || !Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: 'Missing required fields: model, messages[]', type: 'invalid_request_error', code: 400 }
      });
    }

    const requestedModel = String(model).trim();
    const mappedModel = MODEL_MAPPING[requestedModel] || requestedModel;
    const nimModel = normalizeMappedModelId(mappedModel);

    let nimRequest = {
      model: nimModel,
      messages,
      temperature: typeof body.temperature === 'number' ? body.temperature : 0.7,
      top_p: typeof body.top_p === 'number' ? body.top_p : undefined,
      max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024,
      seed: typeof body.seed === 'number' ? body.seed : undefined,
      stream: !!stream
    };

    // forward extra fields if present
    for (const k of ['presence_penalty', 'frequency_penalty', 'stop', 'n']) {
      if (body[k] !== undefined) nimRequest[k] = body[k];
    }

    // forward user-provided extra_body and top-level chat_template_kwargs
    if (body.extra_body && typeof body.extra_body === 'object') {
      nimRequest.extra_body = deepMerge(nimRequest.extra_body || {}, cloneJSON(body.extra_body));
    }
    if (body.chat_template_kwargs && typeof body.chat_template_kwargs === 'object') {
      nimRequest.chat_template_kwargs = deepMerge(nimRequest.chat_template_kwargs || {}, cloneJSON(body.chat_template_kwargs));
    }

    // merges
    if (REQUEST_MERGE_GLOBAL && Object.keys(REQUEST_MERGE_GLOBAL).length) {
      nimRequest = deepMerge(nimRequest, cloneJSON(REQUEST_MERGE_GLOBAL));
    }
    const perModelMerge = getPerModelConfig(REQUEST_MERGE_BY_MODEL, nimModel);
    if (perModelMerge) {
      nimRequest = deepMerge(nimRequest, cloneJSON(perModelMerge));
    }
    if (EXTRA_BODY_GLOBAL && Object.keys(EXTRA_BODY_GLOBAL).length) {
      nimRequest.extra_body = deepMerge(nimRequest.extra_body || {}, cloneJSON(EXTRA_BODY_GLOBAL));
    }
    const perModelExtra = getPerModelConfig(EXTRA_BODY_BY_MODEL, nimModel);
    if (perModelExtra) {
      nimRequest.extra_body = deepMerge(nimRequest.extra_body || {}, cloneJSON(perModelExtra));
    }

    // clean accidental nesting
    if (nimRequest.extra_body && nimRequest.extra_body.extra_body) delete nimRequest.extra_body.extra_body;

    const showReasoning = shouldShowReasoning(nimModel);

    // -------- GLM handling (keep thinking, remove unsupported top-level params) --------
    if (ENABLE_THINKING_MODE && isGlm(nimModel)) {
      // Hoist OpenAI-python style extra_body.chat_template_kwargs -> top-level (GLM prefers top-level)
      if (nimRequest.extra_body?.chat_template_kwargs) {
        nimRequest.chat_template_kwargs = deepMerge(
          nimRequest.chat_template_kwargs || {},
          cloneJSON(nimRequest.extra_body.chat_template_kwargs)
        );
      }

      // Always keep thinking enabled for GLM when thinking mode is on
      // (even if you don’t show reasoning, thinking can still help answer quality)
      nimRequest.chat_template_kwargs = nimRequest.chat_template_kwargs || {};
      if (nimRequest.chat_template_kwargs.enable_thinking === undefined) {
        nimRequest.chat_template_kwargs.enable_thinking = true;
      }
      if (nimRequest.chat_template_kwargs.clear_thinking === undefined) {
        nimRequest.chat_template_kwargs.clear_thinking = GLM_CLEAR_THINKING;
      }

      // CRITICAL FIX: strip keys NVIDIA started rejecting
      stripUnsupportedGlmTopLevelParams(nimRequest);
    }

    // -------- Non-GLM reasoning enablement (DeepSeek etc.) --------
    if (ENABLE_THINKING_MODE && !isGlm(nimModel) && showReasoning) {
      nimRequest.extra_body = nimRequest.extra_body || {};
      nimRequest.extra_body.chat_template_kwargs = nimRequest.extra_body.chat_template_kwargs || {};
      if (nimRequest.extra_body.chat_template_kwargs.thinking === undefined) {
        nimRequest.extra_body.chat_template_kwargs.thinking = true;
      }

      if (nimRequest.enable_reasoning === undefined) nimRequest.enable_reasoning = true;
      if (nimRequest.include_reasoning === undefined) nimRequest.include_reasoning = true;
      if (nimRequest.reasoning === undefined) nimRequest.reasoning = { effort: 'medium' };
    }

    if (DEBUG_REASONING && isGlm(nimModel)) {
      console.log('GLM request body (sanitized):', JSON.stringify(nimRequest, null, 2));
    }

    const axiosConfig = {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      validateStatus: s => s < 600
    };

    const upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, axiosConfig);

    if (upstream.status >= 400) {
      return await sendUpstreamError(res, upstream.status, upstream.data, 'Upstream returned an error');
    }

    const glmReasoningAsContent = showReasoning && reasoningAsContentModel(nimModel);

    // ---------------- STREAM ----------------
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
        if (!showReasoning || glmReasoningAsContent || !reasoningBuf || emittedReasoningBlock) return;
        const block = `${THINK_OPEN}\n${reasoningBuf}\n${THINK_CLOSE}\n\n`;
        emit({
          id: `chunk-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: requestedModel,
          choices: [{ index: 0, delta: { content: block }, finish_reason: null }]
        });
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
                if (glmReasoningAsContent) {
                  delta.content = (delta.content || '') + r;
                } else {
                  reasoningBuf += r;
                  const onlyReasoning = !delta.content || delta.content.length === 0;
                  if (onlyReasoning) continue;
                }
              }
              if (!glmReasoningAsContent && delta.content && !emittedReasoningBlock && reasoningBuf.length) {
                emitReasoningBlockIfNeeded();
              }
            }

            emit(data);
          } catch {
            res.write(line + '\n');
          }
        }
      });

      upstream.data.on('end', () => res.end());
      upstream.data.on('error', () => res.end());
      return;
    }

    // ---------------- NON-STREAM ----------------
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: requestedModel,
      choices: (upstream.data?.choices || []).map((choice, idx) => {
        const role = choice?.message?.role || 'assistant';
        let content = textFromAny(choice?.message?.content || '');

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
    if (error.response) {
      const status = error.response.status || 500;
      return await sendUpstreamError(res, status, error.response.data, error.message || 'Upstream request failed');
    }
    res.status(500).json({ error: { message: error?.message || 'Internal server error', type: 'proxy_error', code: 500 } });
  }
});

app.listen(PORT, () => {
  console.log(`OpenAI->NIM Proxy running on port ${PORT}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`GLM target: ${GLM_UPSTREAM_ID}`);
});
