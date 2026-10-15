//===--- skepu_lsp.h - The SkePU additions to clangd ------------------------===//
//
// The SkePU language server is clangd with these additions. clangd.patch only calls them, from:
//
//   Hover.cpp          getHover():               skepuHover()
//   FindSymbols.cpp    collectDocSymbols():      skepuDocumentSymbols()
//   CodeComplete.cpp   SignatureHelpCollector:   skepuSignatureHelp()
//
// They use what the matcher of skepu-tool (SkepuMatcher) finds in a file, kept by file name as a copy (SkepuInfo).
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
