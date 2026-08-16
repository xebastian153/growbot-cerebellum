// Verifies growbot_forward.js against the trained PyTorch net.
//   node test_forward.mjs
// Passes if every single-step prediction and every step of a 25-tick imagined
// rollout match the reference vectors to float32 precision.

import { readFileSync } from "node:fs";
import { GrowBotForward } from "./growbot_forward.js";

const here = new URL(".", import.meta.url).pathname;
const fwd = new GrowBotForward(JSON.parse(readFileSync(here + "forward_85mm.json", "utf8")));
const ref = JSON.parse(readFileSync(here + "reference_vectors.json", "utf8"));

let worst = 0, n = 0;
for (let i = 0; i < ref.single.x.length; i++) {
  const y = fwd.step(ref.single.x[i]);
  for (let j = 0; j < y.length; j++) { worst = Math.max(worst, Math.abs(y[j] - ref.single.y[i][j])); n++; }
}
console.log(`single-step: ${n} outputs, max |js - torch| = ${worst.toExponential(2)}`);

// rollout: seed the history from the reference, imagine the reference plan
const r = ref.rollout;
fwd.hist = r.hist_imu9.map((f, k) => [...f, ...r.hist_action[k]]);
const im = fwd.imagine(r.plan).imu9;
let worstR = 0;
for (let h = 0; h < im.length; h++)
  for (let j = 0; j < 9; j++) worstR = Math.max(worstR, Math.abs(im[h][j] - r.imagined[h][j]));
console.log(`25-tick rollout: max |js - torch| = ${worstR.toExponential(2)}`);

const TOL = 2e-4;   // float32 accumulation over 25 steps
const ok = worst < TOL && worstR < TOL;
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
