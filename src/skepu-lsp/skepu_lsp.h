//===--- skepu_lsp.h - The SkePU additions to clangd ------------------------===//
//
// The SkePU language server is clangd with these additions. They are in the files of SkePU (this one and
// skepu_lsp.cpp, linked into clang-tools-extra/clangd by the build of SkePU), and clangd.patch only calls them
// from the places in clangd that the results are shown from:
//
//   Hover.cpp          getHover():               skepuHover()
//   FindSymbols.cpp    collectDocSymbols():      skepuDocumentSymbols()
//   CodeComplete.cpp   SignatureHelpCollector:   skepuSignatureHelp()
//
// What they know about a file is found by the matcher of skepu-tool (SkepuMatcher), which runs when it is first needed
// and when the file has changed on the disk. The result is kept by file name (SkepuInfo): it is a copy of what
// is needed, not pointers into the data of the matcher, that the next run of the matcher deletes.
//
//===------------------------------------------------------------------------===//
#pragma once

#include "Protocol.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include <vector>

namespace clang {
class FunctionDecl;
class NamedDecl;
class Sema;
namespace clangd {

class ParsedAST;
struct HoverInfo;

/// Hover on a skeleton (the type, such as skepu::Map, or an instance of it): the documentation and the parameters of the skeleton.
/// \p Function is the function that Decl is (or has the type of), if any.
void skepuHover(ParsedAST &AST, const NamedDecl &Decl, const FunctionDecl *Function, HoverInfo &HI);

/// The document symbols of a file, grouped as SkePU's constructs (skeletons, user functions, constants, types, containers).
std::vector<DocumentSymbol> skepuDocumentSymbols(ParsedAST &AST, std::vector<DocumentSymbol> Syms);

/// Signature help for a call to a skeleton instance (and the skeleton factory functions): the parameters, in terms of SkePU,
/// replace the signatures that clangd has made from the candidates, if any of them is a skeleton.
void skepuSignatureHelp(Sema &S, unsigned CurrentArg, CodeCompleteConsumer::OverloadCandidate *Candidates, unsigned NumCandidates,
                        SignatureHelp &SigHelp);

} // namespace clangd
} // namespace clang
