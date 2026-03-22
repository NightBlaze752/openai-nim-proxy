const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY || '';

const ENABLE_THINKING_MODE =
  String(process.env.ENABLE_THINKING_MODE || 'false').toLowerCase() === 'true';

const DEBUG_REASONING =
  String(process.env.DEBUG_REASONING || 'false').toLowerCase() === 'true';

const SHOW_REASONING_MODELS = (process.env.SHOW_REASONING_MODELS || '')
  .split(',')
  .map((s) => s.trim().toLowerCase())
  .filter(Boolean);

const THINK_OPEN = process.env.THINK_OPEN_TAG || '<think>';
const THINK_CLOSE = process.env.THINK_CLOSE_TAG || '</think>';

const GLM_UPSTREAM_ID = 'z-ai/glm5';

function parseJSONEnv(name) {
  if (!process.env[name]) return null;
  try {
    return JSON.parse(process.env[name]);
  } catch {
    console.warn(`Invalid JSON in ${name}. Ignoring.`);
    return null;
  }
}

function cloneJSON(value) {
  if (value == null) return value;
  return JSON.parse(JSON.stringify(value));
}

function deepMerge(target = {}, source) {
  if (!source || typeof source !== 'object') return target;

  for (const key of Object.keys(source)) {
    const srcVal = source[key];
    const tgtVal = target[key];

    if (srcVal && typeof srcVal === 'object' && !Array.isArray(srcVal)) {
      target[key] = deepMerge(
        tgtVal && typeof tgtVal === 'object' && !Array.isArray(tgtVal) ? tgtVal : {},
        srcVal
      );
    } else {
      target[key] = srcVal;
    }
  }

  return target;
}

function shouldShowReasoning(nimModelId) {
  if (!nimModelId || SHOW_REASONING_MODELS.length === 0) return false;
  const id = String(nimModelId).toLowerCase();
  return SHOW_REASONING_MODELS.some((token) => id.includes(token));
}

function reasoningAsContentModel(nimModelId) {
  const id = String(nimModelId || '').toLowerCase();
  return id === GLM_UPSTREAM_ID || id.includes(GLM_UPSTREAM_ID);
}

function isDeepSeekThinkingModel(nimModelId) {
  const id = String(nimModelId || '').toLowerCase();
  return (
    id === 'deepseek-ai/deepseek-v3.1' ||
    id === 'deepseek-ai/deepseek-v3.2' ||
    id === 'deepseek-ai/deepseek-v3.1-terminus'
  );
}

function textFromAny(v) {
  if (v == null) return '';

  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);

  if (Array.isArray(v)) {
    return v.map(textFromAny).join('');
  }

  if (typeof v === 'object') {
    if (typeof v.text === 'string') return v.text;
    if (typeof v.content === 'string') return v.content;
    if (Array.isArray(v.content)) return v.content.map(textFromAny).join('');
    if (typeof v.reasoning_content === 'string') return v.reasoning_content;
    if (typeof v.reasoning === 'string') return v.reasoning;

    let buf = '';
    for (const val of Object.values(v)) {
      buf += textFromAny(val);
    }
    return buf;
  }

  return '';
}

const GLM_ROOT_REASONING_BUNDLE = {
  reasoning: { effort: 'medium' },
  enable_reasoning: true,
  include_reasoning: true,
  enable_thinking: true,
  thinking: { type: 'enabled' }
};

const GLM_EXTRA_REASONING_BUNDLE = {
  chat_template_kwargs: {
    thinking: true,
    enable_thinking: true
  }
};

// Defaults — override with MODEL_MAP_OVERRIDES
const DEFAULT_MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o-mini': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',

  // DeepSeek aliases
  'deepseek-v3.1': 'deepseek-ai/deepseek-v3.1',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3.2',
  'deepseek-v3.1-terminus': 'deepseek-ai/deepseek-v3.1-terminus',

  // GLM aliases
  'glm5': GLM_UPSTREAM_ID,
  'glm-5': GLM_UPSTREAM_ID
};

let MODEL_MAPPING = { ...DEFAULT_MODEL_MAPPING };
const MODEL_MAP_OVERRIDES = parseJSONEnv('MODEL_MAP_OVERRIDES');
if (MODEL_MAP_OVERRIDES) {
  MODEL_MAPPING = { ...MODEL_MAPPING, ...MODEL_MAP_OVERRIDES };
  console.log('Loaded MODEL_MAP_OVERRIDES:', MODEL_MAPPING);
}

const DEFAULT_REQUEST_MERGE_BY_MODEL = ENABLE_THINKING_MODE
  ? {
      'deepseek-ai/deepseek-v3.1': {
        reasoning: { effort: 'medium' },
        enable_reasoning: true,
        include_reasoning: true
      },
      'deepseek-ai/deepseek-v3.2': {
        reasoning: { effort: 'medium' },
        enable_reasoning: true,
        include_reasoning: true
      },
      'deepseek-ai/deepseek-v3.1-terminus': {
        reasoning: { effort: 'medium' },
        enable_reasoning: true,
        include_reasoning: true
      },
      [GLM_UPSTREAM_ID]: cloneJSON(GLM_ROOT_REASONING_BUNDLE)
    }
  : {};

const DEFAULT_EXTRA_BODY_BY_MODEL = ENABLE_THINKING_MODE
  ? {
      'deepseek-ai/deepseek-v3.1': {
        chat_template_kwargs: { thinking: true }
      },
      'deepseek-ai/deepseek-v3.2': {
        chat_template_kwargs: { thinking: true }
      },
      'deepseek-ai/deepseek-v3.1-terminus': {
        chat_template_kwargs: { thinking: true }
      },
      [GLM_UPSTREAM_ID]: cloneJSON(GLM_EXTRA_REASONING_BUNDLE)
    }
  : {};

const REQUEST_MERGE_GLOBAL = parseJSONEnv('REQUEST_MERGE_GLOBAL') || {};
const REQUEST_MERGE_BY_MODEL = deepMerge(
  cloneJSON(DEFAULT_REQUEST_MERGE_BY_MODEL),
  parseJSONEnv('REQUEST_MERGE_BY_MODEL') || {}
);
const EXTRA_BODY_GLOBAL = parseJSONEnv('EXTRA_BODY_GLOBAL') || {};
const EXTRA_BODY_BY_MODEL = deepMerge(
  cloneJSON(DEFAULT_EXTRA_BODY_BY_MODEL),
  parseJSONEnv('EXTRA_BODY_BY_MODEL') || {}
);

const REASONING_FIELDS = [
  'reasoning_content',
  'reasoning',
  'thoughts',
  'thinking',
  'chain_of_thought',
  'reasoning_text',
  'analysis'
];

function extractReasoningFromDelta(delta) {
  let buf = '';

  for (const f of REASONING_FIELDS) {
    const extracted = textFromAny(delta?.[f]);
    if (extracted) {
      buf += extracted;
      delete delta[f];
    }
  }

  return buf;
}

function extractReasoningFromMessage(msg) {
  if (!msg || typeof msg !== 'object') return '';

  let buf = '';
  for (const f of REASONING_FIELDS) {
    const extracted = textFromAny(msg[f]);
    if (extracted) buf += extracted;
  }

  return buf;
}

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
        return res.status(status).json({
          error: { message: text || fallbackMessage, type: 'upstream_error', code: status }
        });
      }
    }

    if (typeof data === 'string') {
      try {
        return res.status(status).json(JSON.parse(data));
      } catch {
        return res.status(status).json({
          error: { message: data || fallbackMessage, type: 'upstream_error', code: status }
        });
      }
    }

    if (data && typeof data === 'object') {
      return res.status(status).json(data);
    }

    return res.status(status).json({
      error: { message: fallbackMessage, type: 'upstream_error', code: status }
    });
  } catch (e) {
    return res.status(500).json({
      error: {
        message: e?.message || 'Failed to forward upstream error',
        type: 'proxy_error',
        code: 500
      }
    });
  }
}

function getPerModelConfig(map, nimModel) {
  if (!nimModel) return null;
  if (map[nimModel]) return map[nimModel];

  if (nimModel === 'deepseek-ai/deepseek-v3.2' && map['deepseek-ai/deepseek-v3.1']) {
    return map['deepseek-ai/deepseek-v3.1'];
  }

  if (nimModel.startsWith('deepseek-ai/deepseek-v3') && map['deepseek-ai/deepseek-v3']) {
    return map['deepseek-ai/deepseek-v3'];
  }

  return null;
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI->NIM Proxy',
    thinking_mode: ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED',
    show_reasoning_allowlist: SHOW_REASONING_MODELS,
    has_nim_key: !!NIM_API_KEY
  });
});

app.get('/v1/models', async (req, res) => {
  try {
    const r = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` }
    });

    const upstream = r.data?.data || r.data || [];
    const list = Array.isArray(upstream) ? upstream : upstream.data || [];

    const aliases = Object.keys(MODEL_MAPPING).map((id) => ({
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
    res.status(status).json({
      error: { message: msg, type: 'models_error', code: status }
    });
  }
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body || {};

    if (!model || !Array.isArray(messages)) {
      return res.status(400).json({
        error: {
          message: 'Missing required fields: model, messages[]',
          type: 'invalid_request_error',
          code: 400
        }
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

    // Global thinking hint
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = nimRequest.extra_body || {};
      nimRequest.extra_body.chat_template_kwargs = nimRequest.extra_body.chat_template_kwargs || {};
      nimRequest.extra_body.chat_template_kwargs.thinking = true;

      if (nimModel === GLM_UPSTREAM_ID) {
        nimRequest.extra_body.chat_template_kwargs.enable_thinking = true;
      }
    }

    // Top-level merges
    if (REQUEST_MERGE_GLOBAL && Object.keys(REQUEST_MERGE_GLOBAL).length) {
      nimRequest = deepMerge(nimRequest, cloneJSON(REQUEST_MERGE_GLOBAL));
    }

    const reqMergeForModel = getPerModelConfig(REQUEST_MERGE_BY_MODEL, nimModel);
    if (reqMergeForModel) {
      nimRequest = deepMerge(nimRequest, cloneJSON(reqMergeForModel));
    }

    // extra_body merges
    if (EXTRA_BODY_GLOBAL && Object.keys(EXTRA_BODY_GLOBAL).length) {
      nimRequest.extra_body = deepMerge(
        nimRequest.extra_body || {},
        cloneJSON(EXTRA_BODY_GLOBAL)
      );
    }

    const extraForModel = getPerModelConfig(EXTRA_BODY_BY_MODEL, nimModel);
    if (extraForModel) {
      nimRequest.extra_body = deepMerge(
        nimRequest.extra_body || {},
        cloneJSON(extraForModel)
      );
    }

    // Force DeepSeek thinking config after merges so env vars don’t accidentally disable it
    if (ENABLE_THINKING_MODE && isDeepSeekThinkingModel(nimModel)) {
      nimRequest = deepMerge(nimRequest, {
        reasoning: { effort: 'medium' },
        enable_reasoning: true,
        include_reasoning: true
      });

      nimRequest.extra_body = deepMerge(nimRequest.extra_body || {}, {
        chat_template_kwargs: { thinking: true }
      });
    }

    // Force GLM5 thinking config after merges
    if (ENABLE_THINKING_MODE && nimModel === GLM_UPSTREAM_ID) {
      nimRequest = deepMerge(nimRequest, cloneJSON(GLM_ROOT_REASONING_BUNDLE));
      nimRequest.extra_body = deepMerge(
        nimRequest.extra_body || {},
        cloneJSON(GLM_EXTRA_REASONING_BUNDLE)
      );
    }

    if (DEBUG_REASONING && (isDeepSeekThinkingModel(nimModel) || nimModel === GLM_UPSTREAM_ID)) {
      console.log('FINAL NIM REQUEST:', JSON.stringify(nimRequest, null, 2));
    }

    const axiosConfig = {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      validateStatus: (s) => s < 600
    };

    const upstream = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      axiosConfig
    );

    if (upstream.status >= 400) {
      return await sendUpstreamError(res, upstream.status, upstream.data, 'Upstream returned an error');
    }

    const showReasoning = shouldShowReasoning(nimModel);
    const glmReasoningAsContent = showReasoning && reasoningAsContentModel(nimModel);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      let buffer = '';
      let reasoningBuf = '';
      let emittedReasoningBlock = false;
      let loggedFirstChunk = false;

      function emit(obj) {
        res.write(`data: ${JSON.stringify(obj)}\n\n`);
      }

      function emitReasoningBlockIfNeeded() {
        if (!showReasoning || glmReasoningAsContent || !reasoningBuf || emittedReasoningBlock) return;

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

            if (
              DEBUG_REASONING &&
              (isDeepSeekThinkingModel(nimModel) || nimModel === GLM_UPSTREAM_ID) &&
              !loggedFirstChunk
            ) {
              loggedFirstChunk = true;
              console.log('Reasoning stream first chunk keys:', Object.keys(data || {}));
              console.log('Reasoning stream first delta keys:', Object.keys(delta || {}));
              console.log('Reasoning stream first chunk raw:', JSON.stringify(data, null, 2));
            }

            if (delta.content != null && typeof delta.content !== 'string') {
              delta.content = textFromAny(delta.content);
            }

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

            for (const f of REASONING_FIELDS) {
              if (f in delta) delete delta[f];
            }

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

    const upstreamChoices = upstream.data?.choices || [];

    if (
      DEBUG_REASONING &&
      (isDeepSeekThinkingModel(nimModel) || nimModel === GLM_UPSTREAM_ID) &&
      upstreamChoices[0]?.message
    ) {
      console.log('Response message keys:', Object.keys(upstreamChoices[0].message));
      console.log('Raw first choice message:', JSON.stringify(upstreamChoices[0].message, null, 2));
    }

    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: upstreamChoices.map((choice, idx) => {
        const role = choice?.message?.role || 'assistant';
        let content = textFromAny(choice?.message?.content || '');

        if (showReasoning) {
          const r = extractReasoningFromMessage(choice?.message);

          if (r) {
            if (glmReasoningAsContent) {
              content = content ? `${r}\n\n${content}` : r;
            } else {
              content = `${THINK_OPEN}\n${r}\n${THINK_CLOSE}\n\n${content}`;
            }
          }
        }

        return {
          index: choice?.index ?? idx,
          message: { role, content },
          finish_reason: choice?.finish_reason || 'stop'
        };
      }),
      usage: upstream.data?.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);
  } catch (error) {
    if (error.response) {
      const status = error.response.status || 500;
      return await sendUpstreamError(
        res,
        status,
        error.response.data,
        error.message || 'Upstream request failed'
      );
    }

    const safeMsg = error?.message || 'Internal server error';
    console.error('Proxy error:', { status: 500, message: safeMsg });
    res.status(500).json({
      error: { message: safeMsg, type: 'invalid_request_error', code: 500 }
    });
  }
});

app.use((req, res) => {
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
