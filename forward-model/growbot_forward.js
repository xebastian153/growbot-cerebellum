// GrowBot 85mm forward model — pure-JS forward pass. No deps; runs in the browser
// or Node. Verified against the trained PyTorch net (test_forward.mjs).
//
// What it is: the robot's physical imagination. Given the last few IMU readings
// and leg commands, it predicts what the IMU will read one tick (20 ms) later.
// Roll it forward on its own predictions and you get "what happens if I move my
// legs like this" before the legs move. That is the piece the launch video names
// as missing — the cerebellum half that predicts, so the other half can compare
// prediction with reality and learn.
//
// Contract (mirrors the walk policy's):
//   imu6    = [roll, pitch, yaw, gyroRoll, gyroPitch, gyroYaw]  — same frame and
//             sign calibration you already feed GrowBotPolicy.
//   action2 = [aRight, aLeft] in radians of leg swing — the walk policy's output
//             units, before the degrees conversion.
//   The model keeps a 5-deep history of (imu, action) pairs, newest first, and
//   predicts the *change* in the IMU over the next tick. Angles live as (sin, cos)
//   internally so yaw wrap and a roll through ±π in a fall never jump.

const swish = (v) => v / (1 + Math.exp(-v));

export class GrowBotForward {
  constructor(json) {
    this.p = json;                       // forward_85mm.json
    this.K = json.history;               // 5
    this.reset();
  }

  // ---- history ring: newest first, K entries of [imu9..., aRight, aLeft] ----
  reset(imu6) {
    const f = imu6 ? GrowBotForward.encode(imu6) : new Array(9).fill(0);
    this.hist = [];
    for (let k = 0; k < this.K; k++) this.hist.push([...f, 0, 0]);
  }
  // Call once per tick with what the IMU reads now and what you commanded now.
  observe(imu6, action2) {
    this.hist.unshift([...GrowBotForward.encode(imu6), action2[0], action2[1]]);
    this.hist.length = this.K;
  }

  // ---- one-tick prediction: imu9 delta given a flat window (K*11) ----
  step(win) {
    const p = this.p;
    let x = win.map((v, i) => (v - p.in_mean[i]) / p.in_std[i]);
    for (const L of p.layers) {
      const W = L.W, b = L.b, out = new Array(b.length).fill(0);
      for (let i = 0; i < W.length; i++) { const xi = x[i], Wi = W[i]; for (let j = 0; j < Wi.length; j++) out[j] += xi * Wi[j]; }
      for (let j = 0; j < b.length; j++) out[j] += b[j];
      x = (L.act === "swish") ? out.map(swish) : out;
    }
    return x.map((v, i) => v * p.out_std[i] + p.out_mean[i]);
  }

  // ---- imagine: roll a plan of actions forward from the current history ----
  // plan: [[aRight, aLeft], ...]  ->  imagined imu6 per tick, plus the raw imu9s.
  imagine(plan) {
    const K = this.K;
    const win = this.hist.map((r) => r.slice());        // (K, 11) newest first
    let cur = win[0].slice(0, 9);
    const imu9s = [], imu6s = [];
    for (let h = 0; h < plan.length; h++) {
      win[0][9] = plan[h][0]; win[0][10] = plan[h][1];    // this tick's command
      const d = this.step(win.flat());
      cur = cur.map((v, i) => v + d[i]);
      for (let a = 0; a < 3; a++) {                        // keep (sin, cos) on the circle
        const n = Math.hypot(cur[a], cur[a + 3]) + 1e-9; cur[a] /= n; cur[a + 3] /= n;
      }
      imu9s.push(cur.slice()); imu6s.push(GrowBotForward.decode(cur));
      win.unshift([...cur, 0, 0]); win.length = K;
    }
    return { imu6: imu6s, imu9: imu9s };
  }

  static encode(imu6) {
    const [r, p, y, gr, gp, gy] = imu6;
    return [Math.sin(r), Math.sin(p), Math.sin(y), Math.cos(r), Math.cos(p), Math.cos(y), gr, gp, gy];
  }
  static decode(imu9) {
    return [Math.atan2(imu9[0], imu9[3]), Math.atan2(imu9[1], imu9[4]), Math.atan2(imu9[2], imu9[5]),
            imu9[6], imu9[7], imu9[8]];
  }
}

// ---- optional: a small planner that uses the imagination to hit a target ----
// Cross-entropy search over an action chunk. Score = mean squared roll/pitch error
// between imagined and target imu6 traces. ~256 candidates × 4 iterations × H ticks
// of a 25k-param net: a few ms in a phone browser for H = 15.
export function planToMatch(fwd, target6, opts = {}) {
  const H = target6.length, n = opts.n ?? 256, iters = opts.iters ?? 4, elite = opts.elite ?? 32;
  const smooth = opts.smooth ?? 0.6, rand = opts.rand ?? Math.random;
  let mean = opts.init ?? Array.from({ length: H }, () => [0, 0]);
  let std = Array.from({ length: H }, () => [0.6, 0.6]);
  const gauss = () => { let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand();
                        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const wrap = (e) => Math.atan2(Math.sin(e), Math.cos(e));
  for (let it = 0; it < iters; it++) {
    const cands = [];
    for (let c = 0; c < n; c++) {
      const plan = [];
      for (let h = 0; h < H; h++) {
        let a = [mean[h][0] + std[h][0] * gauss(), mean[h][1] + std[h][1] * gauss()];
        if (h) a = [smooth * plan[h - 1][0] + (1 - smooth) * a[0], smooth * plan[h - 1][1] + (1 - smooth) * a[1]];
        plan.push([Math.max(-1.4, Math.min(1.4, a[0])), Math.max(-1.4, Math.min(1.4, a[1]))]);
      }
      const im = fwd.imagine(plan).imu6;
      let cost = 0;
      for (let h = 0; h < H; h++) { const e0 = wrap(im[h][0] - target6[h][0]), e1 = wrap(im[h][1] - target6[h][1]); cost += e0 * e0 + e1 * e1; }
      cands.push({ plan, cost });
    }
    cands.sort((a, b) => a.cost - b.cost);
    const top = cands.slice(0, elite);
    mean = Array.from({ length: H }, (_, h) => [0, 1].map((j) => top.reduce((s, c) => s + c.plan[h][j], 0) / elite));
    std = Array.from({ length: H }, (_, h) => [0, 1].map((j) => {
      const m = mean[h][j]; return Math.sqrt(top.reduce((s, c) => s + (c.plan[h][j] - m) ** 2, 0) / elite) + 0.02; }));
  }
  return mean;   // the action chunk to send, tick by tick
}
