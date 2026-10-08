// Opt-in: npm install ollama@0.6.4 in a temporary directory; copy this file
// there and run OLLAMA_HOST=... node ollama_clients.mjs against a chat model.
import assert from 'node:assert/strict';
import { Ollama } from 'ollama';

const headers = process.env.OLLAMA_API_KEY ? { Authorization: `Bearer ${process.env.OLLAMA_API_KEY}` } : {};
const client = new Ollama({ host: process.env.OLLAMA_HOST || 'http://127.0.0.1:11434', headers });
const listed = await client.list();
const model = process.env.OLLAMA_MODEL || listed.models[0]?.model;
assert(model);
assert((await client.ps()).models.length);
assert((await client.show({ model })).capabilities.length);
for (const stream of [false, true]) {
  for (const method of ['chat', 'generate']) {
    const args = method === 'chat'
      ? { messages: [{ role: 'user', content: 'Say hello in one word.' }] }
      : { prompt: 'Say hello in one word.' };
    const response = await client[method]({ model, stream, think: false, options: { temperature: 0, num_predict: 64 }, ...args });
    const chunks = [];
    if (stream) { for await (const chunk of response) chunks.push(chunk); }
    else chunks.push(response);
    assert.equal(chunks.filter(c => c.done).length, 1);
    assert(chunks.at(-1).done && chunks.at(-1).eval_count > 0);
    const text = chunks.map(c => method === 'chat' ? c.message.content : c.response).join('');
    assert(text.trim());
    console.log(method, stream ? 'stream' : 'json', JSON.stringify(text));
  }
}
await assert.rejects(client.generate({ model, prompt: 'hi', keep_alive: '5m' }), e => e.status_code === 400);
console.log('PASS: JavaScript Ollama client discovery, chat/generate JSON+NDJSON and errors');
