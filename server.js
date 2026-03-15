const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

// 1. Safely parse Environment Variables
const safeJSONParse = (envVarStr, fallback) => {
    try {
        return envVarStr ? JSON.parse(envVarStr) : fallback;
    } catch (error) {
        console.error("Failed to parse JSON for env var:", error.message);
        return fallback;
    }
};

const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const EXTRA_BODY_BY_MODEL = safeJSONParse(process.env.EXTRA_BODY_BY_MODEL, {});
const MODEL_MAP_OVERRIDES = safeJSONParse(process.env.MODEL_MAP_OVERRIDES, {});
const REQUEST_MERGE_BY_MODEL = safeJSONParse(process.env.REQUEST_MERGE_BY_MODEL, {});
const REQUEST_MERGE_GLOBAL = safeJSONParse(process.env.REQUEST_MERGE_GLOBAL, {});
const SHOW_REASONING_MODELS = process.env.SHOW_REASONING_MODELS 
    ? process.env.SHOW_REASONING_MODELS.split(',') 
    : [];

// 2. Health Check Endpoint (Required by Render)
app.get('/', (req, res) => {
    res.send({ status: 'Proxy is running' });
});

// 3. Your Proxy Logic goes here
app.post('/v1/chat/completions', async (req, res) => {
    // Example of how to use the variables
    let model = req.body.model;
    
    // Override model if exists in map
    if (MODEL_MAP_OVERRIDES[model]) {
        model = MODEL_MAP_OVERRIDES[model];
    }

    res.json({ message: "Proxy reached", model_used: model });
});

// 4. Start Server (Must use process.env.PORT for Render)
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
    console.log(`Thinking mode enabled: ${ENABLE_THINKING_MODE}`);
});
