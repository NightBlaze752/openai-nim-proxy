// server.js - OpenAI to NVIDIA NIM API Proxy (Render-friendly)
// Notes:
// - Thinking mode is DISABLED by design (do not enable).
// - You can SHOW reasoning ONLY for allowlisted models via SHOW_REASONING_MODELS env var.
// - Use MODEL_MAP_OVERRIDES to inject your exact NVIDIA model IDs from /v1/models.

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Never enable thinking mode (per your requirement)
const ENABLE_THINKING_MODE = false;

// Allowlist for showing reasoning (comma-separated substrings, matched against resolved NIM model id, case-insensitive)
// Example: SHOW_REASONING_MODELS=deepseek,terminus
const SHOW_REASONING_MODELS = (process.env.SHOW_REASONING_MODELS || '')
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

function shouldShowReasoning(nimModelId) {
  if (!nimModelId || SHOW_REASONING_MODELS.length === 0) return false;
  const id = nimModelId.toLowerCase();
  return SHOW_REASONING_MODELS.some(token => id.includes(token));
}

// Default mapping (keep minimal and realistic; override with MODEL_MAP_OVERRIDES)
const DEFAULT_MODEL_MAPPING = {
  // Replace these targets with the exact model ids you see in GET /v1/models
  // Example placeholders (you MUST update to the exact ids you have):
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  // If Terminus is available, map a second alias, e.g.:
  // 'gpt-4o-mini': 'deepseek-ai/deepseek-v3.1-terminus',

  // Safe general fallbacks:
  'gpt-4': 'meta/llama-3.1-70b-instruct',
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct'
};

// Optional: provide JSON to override any mapping, e.g.:
// MODEL_MAP_OVERRIDES={"gpt-4o":"deepseek-ai/DeepSeek-V3.1-OfficialID","gpt-4o-mini":"deepseek-ai/DeepSeek-V3.1-TerminusID"}
let MODEL_MAPPING = { ...DEFAULT_MODEL_MAPPING };
if (process.env.MODEL_MAP_OVERRIDES) {
  try {
    const overrides = JSON.parse(process.env.MODEL_MAP_OVERRIDES);
    MODEL_MAPPING = { ...MODEL_MAPPING, ...overrides };
    console.log('Loaded MODEL_MAP_OVERRIDES:', MODEL_MAPPING);
  } catch (e) {
    console.warn('Invalid MODEL_MAP_OVERRIDES JSON. Ignoring.');
  }
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    thinking_mode: ENABLE_THINKING_MODE ? 'ENABLED (but not used)' : 'DISABLED',
    show_reasoning_allowlist: SHOW_REASONING_MODELS
  });
});

// Optional passthrough to NVIDIA models list (to see what your key can use)
app.get('/v1/models', async (req, res) => {
  try {
    const r = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` }
    });
    // Also append local aliases to help clients discover what to send to /chat/completions
    const aliases = Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'openai-nim-proxy-alias'
    }));
    res.json({
      object: 'list',
      data: [...aliases, ...(r.data?.data || r.data || [])]
    });
  } catch (error) {
    console.error('Models passthrough error:', error.response?.status, error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.message,
        type: error.response?.data?.error?.type || 'models_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Main proxy
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body || {};

    if (!model || !Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: 'Missing required fields: model, messages[]', type: 'invalid_request_error', code: 400 }
      });
    }

    // Resolve NIM model
    // 1) If alias exists, use it; 2) else try model as-is (maybe you pass real NVIDIA id directly)
    let nimModel = MODEL_MAPPING[model] || model;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: typeof temperature === 'number' ? temperature : 0.6,
      max_tokens: typeof max_tokens === 'number' ? max_tokens : 1024,
      // Never send thinking mode
      stream: !!stream
    };

    const axiosConfig = {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      validateStatus: s => s < 500 // surface 4xx as normal responses so we can pass details through
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, axiosConfig);

    // If NVIDIA returns 4xx, forward it as-is
    if (response.status >= 400) {
      return res.status(response.status).json(response.data);
    }

    // Streaming handling
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const showReasoning = shouldShowReasoning(nimModel);
      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;

          // Handle done signal
          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(5).trim());
            const delta = data?.choices?.[0]?.delta || {};

            // Merge reasoning only when allowed
            if (showReasoning) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              let combined = '';

              if (reasoning && !reasoningStarted) {
                combined += '<think>\n' + reasoning;
                reasoningStarted = true;
              } else if (reasoning) {
                combined += reasoning;
              }

              if (content && reasoningStarted) {
                combined += '</think>\n\n' + content;
                reasoningStarted = false;
              } else if (content) {
                combined += content;
              }

              if (combined) {
                data.choices[0].delta.content = combined;
              }
            } else {
              // Hide reasoning for non-allowlisted models
              if (Object.prototype.hasOwnProperty.call(delta, 'reasoning_content')) {
                delete delta.reasoning_content;
              }
              if (!delta.content) {
                delta.content = '';
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // Pass through any non-JSON lines
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
      return;
    }

    // Non-streaming response
    const showReasoning = shouldShowReasoning(nimModel);
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: (response.data?.choices || []).map((choice) => {
        let content = choice?.message?.content || '';
        const role = choice?.message?.role || 'assistant';

        if (showReasoning && choice?.message?.reasoning_content) {
          content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
        }

        return {
          index: choice.index || 0,
          message: { role, content },
          finish_reason: choice.finish_reason || 'stop'
        };
      }),
      usage: response.data?.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    res.json(openaiResponse);
  } catch (error) {
    // Much better diagnostics
    const status = error.response?.status || 500;
    const data = error.response?.data;
    console.error('Proxy error:', {
      message: error.message,
      status,
      data,
      nim_base: NIM_API_BASE
    });

    res.status(status).json({
      error: {
        message: data?.error?.message || data?.message || error.message || 'Internal server error',
        type: data?.error?.type || 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Reasoning allowlist: ${SHOW_REASONING_MODELS.length ? SHOW_REASONING_MODELS.join(', ') : 'OFF'}`);
});
