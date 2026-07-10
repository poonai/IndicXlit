import http from 'k6/http';
import { check } from 'k6';

const URL = __ENV.URL || 'http://localhost:8000';
const VUS = Number(__ENV.VUS || 256);
const DURATION = __ENV.DURATION || '60s';
const OUTPUT_LEN = Number(__ENV.OUTPUT_LEN || 32);

export const options = {
  vus: VUS,
  duration: DURATION,
  thresholds: {
    http_req_failed: ['rate<0.01'],
  },
};

const WORDS = [
  [3, 22, 7, 4, 9, 4, 8, 2],          // bharat
  [3, 6, 4, 16, 4, 15, 8, 11, 2],     // namaste
  [3, 12, 4, 9, 6, 4, 8, 4, 12, 4, 2],// karnataka
  [3, 23, 17, 16, 19, 10, 8, 11, 9, 2]// computer
];

function payload(inputIds) {
  const inputLen = inputIds.length;
  const mask = Array(OUTPUT_LEN * inputLen).fill(true);
  return JSON.stringify({
    inputs: [
      { name: 'input_ids', shape: [1, inputLen], datatype: 'INT32', data: inputIds },
      { name: 'decoder_input_ids', shape: [1, 1], datatype: 'INT32', data: [2] },
      { name: 'input_lengths', shape: [1, 1], datatype: 'INT32', data: [inputLen] },
      { name: 'decoder_input_lengths', shape: [1, 1], datatype: 'INT32', data: [1] },
      { name: 'request_output_len', shape: [1, 1], datatype: 'INT32', data: [OUTPUT_LEN] },
      { name: 'end_id', shape: [1, 1], datatype: 'INT32', data: [2] },
      { name: 'pad_id', shape: [1, 1], datatype: 'INT32', data: [1] },
      { name: 'beam_width', shape: [1, 1], datatype: 'INT32', data: [5] },
      { name: 'num_return_sequences', shape: [1, 1], datatype: 'INT32', data: [5] },
      { name: 'return_log_probs', shape: [1, 1], datatype: 'BOOL', data: [true] },
      {
        name: 'cross_attention_mask',
        shape: [1, OUTPUT_LEN, inputLen],
        datatype: 'BOOL',
        data: mask,
      },
    ],
    outputs: [
      { name: 'output_ids' },
      { name: 'sequence_length' },
      { name: 'cum_log_probs' },
    ],
  });
}

export default function () {
  const ids = WORDS[__ITER % WORDS.length];
  const res = http.post(`${URL}/v2/models/indicxlit_tensorrt_llm/infer`, payload(ids), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '30s',
  });
  check(res, {
    'status is 200': (r) => r.status === 200,
    'has output_ids': (r) => r.body && r.body.includes('output_ids'),
  });
}

