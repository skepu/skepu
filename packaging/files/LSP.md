# The SkePU language server

`bin/clangd` is clangd with SkePU support. Use it in place of the clangd of the editor.

1. **Generate `compile_commands.json`**, which tells the server how the files are compiled: `make compile-commands` in the directory of the Makefile. It has an entry for
   each program of the Makefile. A new program is added to `PROGS` (or `COMMON_SKEPU_SOURCES`, `COMMON_SOURCES`), and the command is run again.
2. **Point the editor to the server.**
   - Visual Studio Code, with the *clangd* extension: set `clangd.path` to the full path of `bin/clangd` (Settings > search for "clangd path"), and
     turn off the C/C++ extension's IntelliSense if it is installed, as the two disagree.
   - Other editors (Neovim, Emacs, Zed, ...): the command that starts the language server for C++ is the full path of `bin/clangd`.
3. Restart the editor (or the language server) after step 1 and after changing the path.

On a skeleton in a program, the hover shows the type of the instance, the user functions it takes and the signature that the skeleton needs. If nothing is shown,
check that `compile_commands.json` is in the directory that is open in the editor and that the file is one of its entries.
