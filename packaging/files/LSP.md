# The SkePU language server

`bin/clangd` is clangd with SkePU support. Use it in place of the clangd of the editor.

1. **Tell the server how your files are compiled.** In the directory of the Makefile run `make compile-commands`. It writes `compile_commands.json` with an entry for
   each program of the Makefile; add new programs to `PROGS` (or `COMMON_SKEPU_SOURCES`, `COMMON_SOURCES`) and run it again.
2. **Point the editor to the server.**
   - Visual Studio Code, with the *clangd* extension: set `clangd.path` to the full path of `bin/clangd` (Settings > search for "clangd path"), and
     turn off the C/C++ extension's IntelliSense if it is installed, as the two disagree.
   - Other editors (Neovim, Emacs, Zed, ...): the command that starts the language server for C++ is the full path of `bin/clangd`.
3. Restart the editor (or the language server) after step 1 and after changing the path.

In a program you now get the information of the skeleton under the cursor: the type of the instance and the user functions it takes, with the signature that the
skeleton needs. If nothing shows, check that `compile_commands.json` is in the directory that you opened and that the file is one of its entries.
