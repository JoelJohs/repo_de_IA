-- IA/scripts/rnn/init.lua
-- Plugin Neovim: autocompletado de C usando la RNN del proyecto 3-RNN.
-- Activa con: require("rnn").setup()
-- Mapea Ctrl+Space en modo insert a la funcion de completar.
--
-- Por defecto apunta a /home/jojo/IA/3-RNN. Para apuntar a otra ruta,
-- pasar opts.rnn_complete_script / opts.python_bin / opts.server_script /
-- opts.model_path en setup().

local M = {}

M.config = {
    rnn_complete_script = "/home/jojo/develop/academic/IA/3-RNN/scripts/rnn-complete.fish",
    python_bin          = "/home/jojo/develop/academic/IA/.venv/bin/python",
    server_script       = "/home/jojo/develop/academic/IA/3-RNN/src/server_stdio.py",
    model_path          = "/home/jojo/develop/academic/IA/3-RNN/models/rnn_v1.keras",
    max_new             = 60,
    temperature         = 0.4,
    seed                = 42,
    trigger_key         = "<C-Space>",
    filetypes           = { "c", "cpp" },
    use_fish_wrapper    = true,
}

-- Llama al wrapper .fish o directamente al server_stdio.py
local function call_rnn(prefix, cfg)
    local request = vim.fn.json_encode({
        id = 1, method = "complete",
        prefix = prefix,
        max_new = cfg.max_new,
        temperature = cfg.temperature,
        seed = cfg.seed,
    })

    local cmd
    if cfg.use_fish_wrapper and vim.fn.filereadable(cfg.rnn_complete_script) == 1 then
        -- Pasa el prefix como argumento al wrapper
        cmd = { cfg.rnn_complete_script, prefix }
    else
        -- Llama directo al server_stdio
        cmd = { cfg.python_bin, cfg.server_script, cfg.model_path }
    end

    if cfg.use_fish_wrapper and vim.fn.filereadable(cfg.rnn_complete_script) == 1 then
        -- Ejecuta el wrapper, captura stdout
        local result = vim.fn.systemlist(cmd)
        -- Devuelve la primera linea (la continuacion sin newline)
        if #result > 0 then
            return result[1]
        end
        return ""
    else
        -- Llama server_stdio con stdin
        local result = vim.fn.systemlist(cmd, request .. "\n")
        for _, line in ipairs(result) do
            local ok, resp = pcall(vim.fn.json_decode, line)
            if ok and resp.ok then
                return resp.text or ""
            end
        end
        return ""
    end
end

function M.complete()
    local row, col = unpack(vim.api.nvim_win_get_cursor(0))
    local line = vim.api.nvim_get_current_line()
    local prefix = line:sub(1, col)

    if prefix == "" or prefix:match("^%s*$") then
        vim.notify("[rnn] Prompt vacio, nada que completar", vim.log.levels.WARN)
        return
    end

    local text = call_rnn(prefix, M.config)
    text = text:gsub("\r", ""):gsub("\n+$", "")

    if text == "" then
        vim.notify("[rnn] El modelo no devolvio sugerencias", vim.log.levels.WARN)
        return
    end

    -- Inserta al cursor
    vim.api.nvim_buf_set_text(0, row - 1, col, row - 1, col, { text })
    vim.api.nvim_win_set_cursor(0, { row, col + #text })
end

-- Vista previa en una ventana flotante: muestra la sugerencia sin insertarla
local preview_state = { buf = nil, win = nil }

function M.preview()
    local row, col = unpack(vim.api.nvim_win_get_cursor(0))
    local line = vim.api.nvim_get_current_line()
    local prefix = line:sub(1, col)

    if prefix == "" or prefix:match("^%s*$") then
        return
    end

    local text = call_rnn(prefix, M.config)
    text = text:gsub("\r", ""):gsub("\n+$", "")
    if text == "" then return end

    -- Cierra cualquier preview anterior
    if preview_state.win and vim.api.nvim_win_is_valid(preview_state.win) then
        vim.api.nvim_win_close(preview_state.win, true)
    end
    if preview_state.buf and vim.api.nvim_buf_is_valid(preview_state.buf) then
        vim.api.nvim_buf_delete(preview_state.buf, { force = true })
    end

    local buf = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_lines(buf, 0, -1, false, { "// prefix: " .. prefix, "// continue:  " .. text })
    vim.api.nvim_buf_set_option(buf, "filetype", "c")

    local width = math.max(40, math.min(120, #text + 20))
    local height = 2
    local win = vim.api.nvim_open_win(buf, false, {
        relative = "cursor",
        row = 1,
        col = 0,
        width = width,
        height = height,
        style = "minimal",
        border = "rounded",
        title = " RNN preview ",
    })
    preview_state.buf = buf
    preview_state.win = win

    -- Cierra con cualquier tecla
    vim.keymap.set("n", "q", function()
        if preview_state.win and vim.api.nvim_win_is_valid(preview_state.win) then
            vim.api.nvim_win_close(preview_state.win, true)
        end
    end, { buffer = buf })
end

function M.setup(opts)
    M.config = vim.tbl_deep_extend("force", M.config, opts or {})

    -- Comando :RnnComplete y :RnnPreview
    vim.api.nvim_create_user_command("RnnComplete", function() M.complete() end, {})
    vim.api.nvim_create_user_command("RnnPreview", function() M.preview() end, {})

    -- Mapea solo en filetypes de C/C++
    vim.api.nvim_create_autocmd("FileType", {
        pattern = M.config.filetypes,
        callback = function()
            vim.keymap.set("i", M.config.trigger_key, function()
                M.complete()
            end, { buffer = true, desc = "RNN: completar" })
            vim.keymap.set("i", "<C-S-p>", function()
                M.preview()
            end, { buffer = true, desc = "RNN: ver preview" })
        end,
    })
end

return M
