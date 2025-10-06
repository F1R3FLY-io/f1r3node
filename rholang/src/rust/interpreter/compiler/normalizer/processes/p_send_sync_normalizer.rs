use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use std::collections::HashMap;
use uuid::Uuid;

use rholang_parser::ast::{AnnName, AnnProc, Bind, Id, SendType, SyncSendCont};

pub fn normalize_p_send_sync_new_ast<'ast>(
    channel: &'ast AnnName<'ast>,
    messages: &'ast rholang_parser::ast::ProcList<'ast>,
    cont: &SyncSendCont<'ast>,
    span: &rholang_parser::SourceSpan,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let identifier = Uuid::new_v4().to_string();

    // Create variable name for the response channel
    // TODO: Replace Box::leak with proper arena allocation for Name
    let name_var = Box::leak(Box::new(rholang_parser::ast::Name::ProcVar(
        rholang_parser::ast::Var::Id(Id {
            // TODO: Replace Box::leak with proper arena allocation for strings
            name: Box::leak(identifier.clone().into_boxed_str()),
            pos: span.start,
        }),
    )));

    // Build the send process: channel!(name_var, ...messages)
    let send: AnnProc = {
        let mut listproc = Vec::new();

        // Add the response channel name as first argument
        listproc.push(AnnProc {
            proc: parser.ast_builder().alloc_eval(AnnName {
                name: *name_var,
                span: *span,
            }),
            span: *span,
        });

        // Add the original messages
        for msg in messages.iter() {
            listproc.push(*msg);
        }

        AnnProc {
            proc: parser
                .ast_builder()
                .alloc_send(SendType::Single, *channel, &listproc),
            span: *span,
        }
    };

    // Build the receive process: for (_ <- name_var) { cont }
    let receive: AnnProc = {
        // Create wildcard pattern
        let wildcard = AnnName {
            name: rholang_parser::ast::Name::ProcVar(rholang_parser::ast::Var::Wildcard),
            span: *span,
        };

        // Create bind for the pattern: _ <- name_var
        let bind = Bind::Linear {
            lhs: rholang_parser::ast::Names {
                names: smallvec::SmallVec::from_vec(vec![wildcard]),
                remainder: None,
            },
            rhs: rholang_parser::ast::Source::Simple {
                name: AnnName {
                    name: *name_var,
                    span: *span,
                },
            },
        };

        // Create receipt containing the bind
        let receipt: smallvec::SmallVec<[Bind<'ast>; 1]> = smallvec::SmallVec::from_vec(vec![bind]);
        let receipts: smallvec::SmallVec<[smallvec::SmallVec<[Bind<'ast>; 1]>; 1]> =
            smallvec::SmallVec::from_vec(vec![receipt]);

        // Get the continuation process
        let cont_proc = match cont {
            SyncSendCont::Empty => AnnProc {
                proc: parser.ast_builder().const_nil(),
                span: *span,
            },
            SyncSendCont::NonEmpty(proc) => *proc,
        };

        AnnProc {
            proc: parser.ast_builder().alloc_for(receipts, cont_proc),
            span: *span,
        }
    };

    // Create name declaration for the new variable
    let name_decl = rholang_parser::ast::NameDecl {
        id: Id {
            // TODO: Replace Box::leak with proper arena allocation for strings
            name: Box::leak(identifier.into_boxed_str()),
            pos: span.start,
        },
        uri: None,
    };

    // Build Par of send and receive
    let p_par = AnnProc {
        proc: parser.ast_builder().alloc_par(send, receive),
        span: *span,
    };

    // Build New process: new name_var in { send | receive }
    let p_new = AnnProc {
        proc: parser.ast_builder().alloc_new(p_par, vec![name_decl]),
        span: *span,
    };

    normalize_ann_proc(&p_new, input, env, parser)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::compiler::exports::{BoundMapChainSpan, FreeMapSpan};
    use crate::rust::interpreter::compiler::normalize::VarSort;
    use models::rhoapi::Par;

    #[test]
    fn p_send_sync_should_normalize_a_basic_send_sync() {
        use rholang_parser::ast::{AnnName, Name, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        fn inputs() -> ProcVisitInputsSpan {
            ProcVisitInputsSpan {
                par: Par::default(),
                bound_map_chain: BoundMapChainSpan::new(),
                free_map: FreeMapSpan::<VarSort>::new(),
            }
        }

        let env = HashMap::<String, Par>::new();
        let parser = rholang_parser::RholangParser::new();

        let channel = AnnName {
            name: Name::ProcVar(Var::Wildcard),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        };

        let messages = smallvec::SmallVec::new();

        let cont = rholang_parser::ast::SyncSendCont::Empty;

        let span = SourceSpan {
            start: SourcePos { line: 3, col: 3 },
            end: SourcePos { line: 3, col: 3 },
        };

        let result = normalize_p_send_sync_new_ast(
            &channel,
            &messages,
            &cont,
            &span,
            inputs(),
            &env,
            &parser,
        );
        assert!(result.is_ok());
    }
}
