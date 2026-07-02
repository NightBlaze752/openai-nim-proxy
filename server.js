const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '4mb' }));

const NIM_API_BASE = String(process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1')
  .trim()
  .replace(/\/+$/, '');

const NIM_API_KEY = process.env.NIM_API_KEY || '';

const ENABLE_THINKING_MODE =
  process.env.ENABLE_THINKING_MODE == null
    ? true
    : String(process.env.ENABLE_THINKING_MODE).toLowerCase() === 'true';

const DEBUG_REASONING = String(process.env.DEBUG_REASONING || 'false').toLowerCase() === 'true';

// Substring allowlist: if nim model id contains any token, we show reasoning
const SHOW_REASONING_MODELS = (process.env.SHOW_REASONING_MODELS || 'glm,v4-pro,deepseek-v3.1,deepseek-v3.2,terminus,r1')
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

const THINK_OPEN = process.env.THINK_OPEN_TAG || '<think>';
const THINK_CLOSE = process.env.THINK_CLOSE_TAG || '</think>';

// ---- GLM 5.2 ----
const GLM_UPSTREAM_ID = 'z-ai/glm-5.2';

// default matches NVIDIA examples (clear_thinking false)
const GLM_CLEAR_THINKING =
  process.env.GLM_CLEAR_THINKING == null
    ? false
    : String(process.env.GLM_CLEAR_THINKING).toLowerCase() === 'true';

// If true: reasoning is mixed directly into content while streaming (always visible)
// If false: reasoning is streamed inside <think>...</think> blocks (cleaner)
const GLM_REASONING_AS_CONTENT = String(process.env.GLM_REASONING_AS_CONTENT || 'false').toLowerCase() === 'true';

// ---------------- helpers ----------------
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

function shouldShowReasoning(nimModelId) {
  if (!nimModelId || SHOW_REASONING_MODELS.length === 0) return false;
  const id = String(nimModelId).toLowerCase();
  return SHOW_REASONING_MODELS.some(token => id.includes(token));
}

function normalizeMappedModelId(modelId) {
  const id = String(modelId || '').trim().toLowerCase();

  // Any older GLM ids -> GLM-5.2
  if (
    id === 'z-ai/glm5' || id === 'z-ai/glm4.7' || id === 'z-ai/glm-5.1' ||
    id === 'glm5' || id === 'glm-5' || id === 'glm5.0' || id === 'glm-5.0' ||
    id === 'glm4.7' || id === 'glm-4.7' ||
    id === 'glm5.1' || id === 'glm-5.1'
  ) return GLM_UPSTREAM_ID;

  // GLM-5.2 direct / alias
  if (id === 'z-ai/glm-5.2' || id === 'glm5.2' || id === 'glm-5.2') return GLM_UPSTREAM_ID;

  return modelId;
}

// Reasoning fields we try to extract
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
  if (typeof v === 'object') return Object.values(v).map(textFromAny).filter(Boolean).join('');
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

// Upstream error forwarding
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
      return res.status(status).json({ error: { message: data || fallbackMessage, type: 'upstream_error', code: status } });
    }
    if (data && typeof data === 'object') return res.status(status).json(data);
    return res.status(status).json({ error: { message: fallbackMessage, type: 'upstream_error', code: status } });
  } catch (e) {
    return res.status(500).json({ error: { message: e?.message || 'Failed to forward upstream error', type: 'proxy_error', code: 500 } });
  }
}

// NVIDIA started rejecting these top-level keys for GLM, keep stripping them:
function stripUnsupportedGlmTopLevelParams(nimRequest) {
  delete nimRequest.enable_reasoning;
  delete nimRequest.enable_thinking;
  delete nimRequest.include_reasoning;
  delete nimRequest.reasoning;
}

// ---------------- model mapping ----------------
const DEFAULT_MODEL_MAPPING = {
  // DeepSeek convenience
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v3.1': 'deepseek-ai/deepseek-v3.1',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',
  'deepseek-v3.1-terminus': 'deepseek-ai/deepseek-v3.1-terminus',
  'deepseek-r1': 'deepseek-ai/deepseek-r1-0528',

  // GLM convenience (map ALL older aliases to 5.2)
  'glm5.2': GLM_UPSTREAM_ID,
  'glm-5.2': GLM_UPSTREAM_ID,
  'glm5.1': GLM_UPSTREAM_ID,
  'glm-5.1': GLM_UPSTREAM_ID,
  'glm5': GLM_UPSTREAM_ID,
  'glm-5': GLM_UPSTREAM_ID,
  'glm5.0': GLM_UPSTREAM_ID,
  'glm-5.0': GLM_UPSTREAM_ID,
  'glm4.7': GLM_UPSTREAM_ID,
  'glm-4.7': GLM_UPSTREAM_ID
};

let MODEL_MAPPING = { ...DEFAULT_MODEL_MAPPING };
const MODEL_MAP_OVERRIDES = parseJSONEnv('MODEL_MAP_OVERRIDES');
if (MODEL_MAP_OVERRIDES && typeof MODEL_MAP_OVERRIDES === 'object') {
  MODEL_MAPPING = { ...MODEL_MAPPING, ...MODEL_MAP_OVERRIDES };
}

// Optional env merges
const REQUEST_MERGE_GLOBAL = parseJSONEnv('REQUEST_MERGE_GLOBAL') || { top_k: -1 };
const REQUEST_MERGE_BY_MODEL = parseJSONEnv('REQUEST_MERGE_BY_MODEL') || {};
const EXTRA_BODY_BY_MODEL = parseJSONEnv('EXTRA_BODY_BY_MODEL') || {};

function getPerModelConfig(map, nimModel) {
  if (!nimModel) return null;
  return map[nimModel] || null;
}

// ---------------- routes ----------------
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

    // Forward client-provided extra_body and chat_template_kwargs
    if (body.extra_body && typeof body.extra_body === 'object') nimRequest.extra_body = cloneJSON(body.extra_body);
    if (body.chat_template_kwargs && typeof body.chat_template_kwargs === 'object') nimRequest.chat_template_kwargs = cloneJSON(body.chat_template_kwargs);

    // Apply global merge
    nimRequest = deepMerge(nimRequest, cloneJSON(REQUEST_MERGE_GLOBAL));

    // Per-model merge (top-level)
    const perModelMerge = getPerModelConfig(REQUEST_MERGE_BY_MODEL, nimModel);
    if (perModelMerge) nimRequest = deepMerge(nimRequest, cloneJSON(perModelMerge));

    // Per-model extra_body merge
    const extraForModel = getPerModelConfig(EXTRA_BODY_BY_MODEL, nimModel);
    if (extraForModel) {
      nimRequest.extra_body = deepMerge(nimRequest.extra_body || {}, cloneJSON(extraForModel));
    }

    const showReasoning = shouldShowReasoning(nimModel);

    // ---- GLM 5.2: thinking via extra_body.chat_template_kwargs (as NVIDIA docs show),
    //      but also hoist to top-level chat_template_kwargs for compatibility.
    if (ENABLE_THINKING_MODE && isGlm(nimModel)) {
      nimRequest.extra_body = nimRequest.extra_body || {};
      nimRequest.extra_body.chat_template_kwargs = nimRequest.extra_body.chat_template_kwargs || {};

      if (nimRequest.extra_body.chat_template_kwargs.enable_thinking === undefined) {
        nimRequest.extra_body.chat_template_kwargs.enable_thinking = true;
      }
      if (nimRequest.extra_body.chat_template_kwargs.clear_thinking === undefined) {
        nimRequest.extra_body.chat_template_kwargs.clear_thinking = GLM_CLEAR_THINKING;
      }

      // Hoist to top-level as well (safe)
      nimRequest.chat_template_kwargs = deepMerge(
        nimRequest.chat_template_kwargs || {},
        cloneJSON(nimRequest.extra_body.chat_template_kwargs)
      );

      // Avoid NVIDIA validation errors
      stripUnsupportedGlmTopLevelParams(nimRequest);
    }

    if (DEBUG_REASONING && isGlm(nimModel)) {
      console.log('GLM request to upstream:', JSON.stringify(nimRequest, null, 2));
    }

    const upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 0,
      validateStatus: s => s < 600
    });

    if (upstream.status >= 400) {
      return await sendUpstreamError(res, upstream.status, upstream.data, 'Upstream returned an error');
    }

    // ---- STREAMING: stream reasoning immediately (no long buffering stalls)
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      let buffer = '';
      let thinkOpenSent = false;
      let thinkClosedSent = false;

      function emit(obj) {
        res.write(`data: ${JSON.stringify(obj)}\n\n`);
      }

      upstream.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();

          if (payload === '[DONE]') {
            if (showReasoning && thinkOpenSent && !thinkClosedSent && !GLM_REASONING_AS_CONTENT) {
              emit({
                id: `chunk-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model: requestedModel,
                choices: [{ index: 0, delta: { content: `\n${THINK_CLOSE}\n\n` }, finish_reason: null }]
              });
              thinkClosedSent = true;
            }
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(payload);
            const delta = data?.choices?.[0]?.delta || {};

            const originalContent = typeof delta.content === 'string'
              ? delta.content
              : (delta.content == null ? '' : textFromAny(delta.content));

            let reasoningText = '';
            if (showReasoning) reasoningText = extractReasoningFromDelta(delta);

            if (showReasoning && reasoningText) {
              if (isGlm(nimModel) && GLM_REASONING_AS_CONTENT) {
                delta.content = (originalContent || '') + reasoningText;
              } else {
                // stream <think> progressively
                let out = '';
                if (!thinkOpenSent) {
                  out += `${THINK_OPEN}\n`;
                  thinkOpenSent = true;
                }

                if (originalContent) {
                  if (!thinkClosedSent) {
                    out += reasoningText + `\n${THINK_CLOSE}\n\n` + originalContent;
                    thinkClosedSent = true;
                  } else {
                    out += reasoningText + originalContent;
                  }
                } else {
                  out += reasoningText;
                }
                delta.content = out;
              }
            } else {
              // no reasoning in this chunk
              if (showReasoning && thinkOpenSent && !thinkClosedSent && originalContent && !(isGlm(nimModel) && GLM_REASONING_AS_CONTENT)) {
                delta.content = `\n${THINK_CLOSE}\n\n` + originalContent;
                thinkClosedSent = true;
              } else {
                delta.content = originalContent;
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

    // Non-streaming passthrough
    return res.json(upstream.data);
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
  console.log(`GLM target: ${GLM_UPSTREAM_ID}`);
});
