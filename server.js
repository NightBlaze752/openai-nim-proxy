// server-thinking.js — Same proxy, but sends thinking=true to models that support it.
// WARNING: Only use if your platform/policy allows it. Some models may error.
// Differences vs server.js:
// - ENABLE_THINKING_MODE = true
// - Adds extra_body: { chat_template_kwargs: { thinking: true } }

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY || '';

const ENABLE_THINKING_MODE = true; // ON in this file

const SHOW_REASONING_MODELS = (process.env.SHOW_REASONING_MODELS || '')
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

const THINK_OPEN = process.env.THINK_OPEN_TAG || '<think>';
const THINK_CLOSE = process.env.THINK_CLOSE_TAG || '</think>';

function shouldShowReasoning(nimModelId) {
  if (!nimModelId || SHOW_REASONING_MODELS.length === 0) return false;
  const id = nimModelId.toLowerCase();
  return SHOW_REASONING_MODELS.some(token => id.includes(token));
}

const DEFAULT_MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o-mini': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4': 'deepseek-ai/deepseek-r1-0528',
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct'
};

let MODEL_MAPPING = { ...DEFAULT_MODEL_MAPPING };
if (process.env.MODEL_MAP_OVERRIDES) {
  try {
    const overrides = JSON.parse(process.env.MODEL_MAP_OVERRIDES);
    MODEL_MAPPING = { ...MODEL_MAPPING, ...overrides };
    console.log('Loaded MODEL_MAP_OVERRIDES:', MODEL_MAPPING);
  } catch {
    console.warn('Invalid MODEL_MAP_OVERRIDES JSON. Ignoring.');
  }
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI->NIM Proxy (thinking)',
    thinking_mode: 'ENABLED',
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
    const msg = (typeof raw === 'string' && raw)
      || raw?.error?.message
      || raw?.message
      || error.message
      || 'models error';
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

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: typeof temperature === 'number' ? temperature : 0.6,
      max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
      stream: !!stream,
      extra_body: { chat_template_kwargs: { thinking: true } } // key difference
    };

    const axiosConfig = {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      validateStatus: s => s < 500
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, axiosConfig);

    if (response.status >= 400) {
      return res.status(response.status).json(response.data);
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

      response.data.on('data', (chunk) => {
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
              if (typeof delta.reasoning_content === 'string' && delta.reasoning_content.length) {
                reasoningBuf += delta.reasoning_content;
                delete delta.reasoning_content;
                const onlyReasoning = !delta.content || delta.content.length === 0;
                if (onlyReasoning) continue;
              }
              if (delta.content && !emittedReasoningBlock && reasoningBuf.length) {
                emitReasoningBlockIfNeeded();
              }
            }

            if ('reasoning_content' in delta) delete delta.reasoning_content;
            emit(data);
          } catch {
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
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
      choices: (response.data?.choices || []).map((choice, idx) => {
        const role = choice?.message?.role || 'assistant';
        let content = choice?.message?.content || '';
        if (showReasoning && choice?.message?.reasoning_content) {
          content = `${THINK_OPEN}\n${choice.message.reasoning_content}\n${THINK_CLOSE}\n\n${content}`;
        }
        return {
          index: choice?.index ?? idx,
          message: { role, content },
          finish_reason: choice?.finish_reason || 'stop'
        };
      }),
      usage: response.data?.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
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
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI->NIM Proxy (thinking) running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
  console.log(`Thinking mode: ENABLED`);
  console.log(`Reasoning allowlist: ${SHOW_REASONING_MODELS.length ? SHOW_REASONING_MODELS.join(', ') : 'OFF'}`);
});
