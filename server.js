ENABLE_THINKING_MODE true

EXTRA_BODY_BY_MODEL {
  "deepseek-ai/deepseek-v3.1": {
    "chat_template_kwargs": { "thinking": true }
  },
  "deepseek-ai/deepseek-v3.2": {
    "chat_template_kwargs": { "thinking": true }
  },
  "deepseek-ai/deepseek-v3.1-terminus": {
    "chat_template_kwargs": { "thinking": true }
  },
  "z-ai/glm5": {
    "chat_template_kwargs": { "thinking": true }
  }
}

MODEL_MAP_OVERRIDES {
  "gpt-4o": "deepseek-ai/deepseek-v3.1",
  "gpt-4o-mini": "deepseek-ai/deepseek-v3.1-terminus",
  "gpt-4": "deepseek-ai/deepseek-r1-0528",

  "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",

  "glm5": "z-ai/glm5",
  "glm-5": "z-ai/glm5",
  "glm4.7": "z-ai/glm5",
  "glm-4.7": "z-ai/glm5"
}

REQUEST_MERGE_BY_MODEL {
  "deepseek-ai/deepseek-v3.1": {
    "reasoning": { "effort": "medium" },
    "enable_reasoning": true,
    "include_reasoning": true,
    "chat_template_kwargs": { "thinking": true }
  },
  "deepseek-ai/deepseek-v3.2": {
    "reasoning": { "effort": "medium" },
    "enable_reasoning": true,
    "include_reasoning": true,
    "chat_template_kwargs": { "thinking": true }
  },
  "deepseek-ai/deepseek-v3.1-terminus": {
    "reasoning": { "effort": "medium" },
    "enable_reasoning": true,
    "include_reasoning": true,
    "chat_template_kwargs": { "thinking": true }
  },
  "z-ai/glm5": {
    "reasoning": { "effort": "medium" },
    "enable_reasoning": true,
    "include_reasoning": true,
    "chat_template_kwargs": { "thinking": true }
  }
}

REQUEST_MERGE_GLOBAL {"top_k":-1}

SHOW_REASONING_MODELS deepseek,terminus,r1,glm
