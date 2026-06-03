/**
 * VS Code extension: char-level vanilla RNN autocompletion for C files.
 *
 * Architecture:
 *   editor  --(JSON line over stdin)-->  python src/server_stdio.py
 *   editor  <--(JSON line on stdout)--  python src/server_stdio.py
 *
 * The server is started once per workspace and reused for every command.
 * Press Ctrl+Shift+Space in a C file to insert the generated continuation.
 */

const { spawn } = require("child_process");
const path = require("path");
const vscode = require("vscode");

let serverProc = null;
let nextId = 1;
const pending = new Map();
let buffer = "";

function workspaceRootPath() {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || !folders.length) {
    return undefined;
  }
  return folders[0].uri.fsPath;
}

function pythonBin() {
  return vscode.workspace.getConfiguration("rnnC").get("pythonPath", "python");
}

function resolveServerArgs() {
  const cfg = vscode.workspace.getConfiguration("rnnC");
  const script = cfg.get("serverScript", "src/server_stdio.py");
  const model = cfg.get("modelPath", "models/rnn_v1.keras");
  const root = workspaceRootPath();
  const scriptAbs = path.isAbsolute(script) ? script : path.join(root || "", script);
  const args = [scriptAbs];
  if (model) {
    const modelAbs = path.isAbsolute(model) ? model : path.join(root || "", model);
    args.push(modelAbs);
  }
  return args;
}

function startServer() {
  if (serverProc && serverProc.exitCode === null) {
    return;
  }
  const py = pythonBin();
  const args = resolveServerArgs();
  console.log(`[rnnC] spawning: ${py} ${args.join(" ")}`);
  serverProc = spawn(py, args, { stdio: ["pipe", "pipe", "pipe"] });

  serverProc.stdout.on("data", (chunk) => {
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
        // ignore malformed
      }
    }
  });

  serverProc.stderr.on("data", (chunk) => {
    // Surface only the tail so the user is not spammed with TF warnings.
    const text = chunk.toString("utf8");
    const tail = text.split("\n").slice(-2).join("\n");
    if (tail.trim()) {
      console.warn("[rnnC server stderr]", tail);
    }
  });

  serverProc.on("exit", (code) => {
    serverProc = null;
    for (const [id, cb] of pending) {
      cb({ ok: false, error: `server exited (code=${code})` });
    }
    pending.clear();
  });
}

function request(method, params, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    startServer();
    if (!serverProc) {
      reject(new Error("server not running"));
      return;
    }
    const id = nextId++;
    const msg = { id, method, ...params };
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`timeout on ${method}`));
    }, timeoutMs);
    pending.set(id, (resp) => {
      clearTimeout(timer);
      if (resp.ok) resolve(resp);
      else reject(new Error(resp.error || "unknown error"));
    });
    serverProc.stdin.write(JSON.stringify(msg) + "\n");
  });
}

function linePrefix(editor) {
  const pos = editor.selection.active;
  return editor.document.getText(
    new vscode.Range(new vscode.Position(pos.line, 0), pos)
  );
}

async function cmdComplete() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;
  const cfg = vscode.workspace.getConfiguration("rnnC");
  const prefix = linePrefix(editor);
  try {
    const r = await request("complete", {
      prefix,
      max_new: cfg.get("maxNew", 60),
      temperature: cfg.get("temperature", 0.4),
    });
    await editor.edit((b) => b.insert(editor.selection.active, r.text));
  } catch (err) {
    vscode.window.showErrorMessage(`RNN complete failed: ${err.message}`);
  }
}

async function cmdSuggest() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;
  const prefix = linePrefix(editor);
  try {
    const r = await request("suggest", { prefix, n: 5 });
    const picked = await vscode.window.showQuickPick(r.items, {
      placeHolder: "RNN: elige el siguiente caracter",
    });
    if (picked) {
      await editor.edit((b) => b.insert(editor.selection.active, picked));
    }
  } catch (err) {
    vscode.window.showErrorMessage(`RNN suggest failed: ${err.message}`);
  }
}

async function cmdStatus() {
  startServer();
  if (!serverProc) {
    vscode.window.showErrorMessage("RNN server not running");
    return;
  }
  try {
    const r = await request("ping", {}, 5000);
    vscode.window.showInformationMessage(
      `RNN server alive (pong=${r.pong}). Use Ctrl+Shift+Space in a C file.`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`RNN ping failed: ${err.message}`);
  }
}

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("rnnC.complete", cmdComplete),
    vscode.commands.registerCommand("rnnC.suggest", cmdSuggest),
    vscode.commands.registerCommand("rnnC.status", cmdStatus)
  );
  startServer();
}

function deactivate() {
  if (serverProc) {
    try {
      serverProc.stdin.end();
    } catch (e) {}
    try {
      serverProc.kill();
    } catch (e) {}
  }
}

module.exports = { activate, deactivate };
