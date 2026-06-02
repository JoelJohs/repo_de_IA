// E2E test: load extension.js, drive it through the same code path VS Code uses.
// We mock vscode and intercept the `request()` flow by re-implementing the
// stdin/stdout protocol against the spawned server.

const { spawn } = require("child_process");
const path = require("path");

const root = path.resolve(__dirname, "..");
const pyBin = "python";
const serverArgs = [
  path.join(root, "src", "server_stdio.py"),
  path.join(root, "models", "rnn_v1.keras"),
];

console.log("Spawning:", pyBin, serverArgs.join(" "));
const proc = spawn(pyBin, serverArgs, {
  cwd: root,
  stdio: ["pipe", "pipe", "pipe"],
});

let nextId = 1;
const pending = new Map();
let buffer = "";

proc.stderr.on("data", (c) => {
  // tail only, like the extension does
  const text = c.toString("utf8");
  const tail = text.split("\n").slice(-1).join("");
  if (tail.trim()) process.stderr.write(`[server] ${tail}`);
});

proc.stdout.on("data", (chunk) => {
  buffer += chunk.toString("utf8");
  let nl;
  while ((nl = buffer.indexOf("\n")) !== -1) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (!line) continue;
    try {
      const msg = JSON.parse(line);
      const cb = pending.get(msg.id);
      if (cb) {
        pending.delete(msg.id);
        cb(msg);
      }
    } catch (e) {
      console.error("bad JSON:", line);
    }
  }
});

proc.on("exit", (code) => {
  console.log(`server exited with code=${code}`);
  for (const [id, cb] of pending) cb({ ok: false, error: "server exited" });
  pending.clear();
});

function request(method, params, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    const msg = { id, method, ...params };
    const t = setTimeout(() => {
      pending.delete(id);
      reject(new Error("timeout"));
    }, timeoutMs);
    pending.set(id, (resp) => {
      clearTimeout(t);
      if (resp.ok) resolve(resp);
      else reject(new Error(resp.error || "unknown"));
    });
    proc.stdin.write(JSON.stringify(msg) + "\n");
  });
}

(async () => {
  try {
    console.log("\n--- ping ---");
    const p = await request("ping", {});
    console.log(">", p);

    console.log("\n--- suggest 'int ' ---");
    const s = await request("suggest", { prefix: "int ", n: 5 });
    console.log(">", s);

    console.log("\n--- complete 'int sum' (max_new=20) ---");
    const c = await request("complete", {
      prefix: "int sum",
      max_new: 20,
      temperature: 0.4,
      seed: 42,
    });
    console.log(">", c);
    console.log(">>> prompt+text:", "int sum" + c.text);

    console.log("\n--- complete 'void print' (max_new=20) ---");
    const c2 = await request("complete", {
      prefix: "void print",
      max_new: 20,
      temperature: 0.4,
      seed: 42,
    });
    console.log(">", c2);
    console.log(">>> prompt+text:", "void print" + c2.text);
  } catch (e) {
    console.error("ERROR:", e.message);
  } finally {
    proc.stdin.end();
    setTimeout(() => process.exit(0), 1000);
  }
})();
