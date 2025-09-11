use crate::rust::interpreter::compiler::exports::{ProcVisitInputsSpan, ProcVisitOutputsSpan};
use crate::rust::interpreter::compiler::normalize::{
    normalize_ann_proc, normalize_match_proc, ProcVisitInputs, ProcVisitOutputs,
};
use crate::rust::interpreter::compiler::rholang_ast;
use crate::rust::interpreter::compiler::rholang_ast::{Name, Proc, SyncSendCont};
use crate::rust::interpreter::compiler::rholang_ast::{NameDecl, ProcList};
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use rholang_parser::ast::{AnnName, AnnProc};
use std::collections::HashMap;
use uuid::Uuid;

pub fn normalize_p_send_sync(
    name: &Name,
    messages: &ProcList,
    cont: &SyncSendCont,
    line_num: usize,
    col_num: usize,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let identifier = Uuid::new_v4().to_string();
    let name_var: rholang_ast::Name =
        Name::ProcVar(Box::new(rholang_ast::Proc::Var(rholang_ast::Var {
            name: identifier.clone(),
            line_num,
            col_num,
        })));

    let send: Proc = {
        let mut listproc = messages.procs.clone();

        listproc.insert(
            0,
            Proc::Eval(rholang_ast::Eval {
                name: name_var.clone(),
                line_num,
                col_num,
            }),
        );

        Proc::Send {
            name: name.clone(),
            send_type: rholang_ast::SendType::Single { line_num, col_num },
            inputs: ProcList {
                procs: listproc,
                line_num,
                col_num,
            },
            line_num: messages.line_num,
            col_num: messages.col_num,
        }
    };

    let receive: Proc = {
        let list_name = rholang_ast::Names {
            names: vec![rholang_ast::Name::ProcVar(Box::new(
                rholang_ast::Proc::Wildcard { line_num, col_num },
            ))],
            cont: None,
            line_num,
            col_num,
        };

        let linear_bind_impl: rholang_ast::LinearBind = rholang_ast::LinearBind {
            names: list_name,
            input: rholang_ast::Source::Simple {
                name: name_var,
                line_num,
                col_num,
            },
            line_num,
            col_num,
        };
        let list_linear_bind = vec![rholang_ast::Receipt::LinearBinds(linear_bind_impl)];

        let list_receipt = rholang_ast::Receipts {
            receipts: list_linear_bind,
            line_num,
            col_num,
        };

        let proc: Box<rholang_ast::Block> = match cont {
            rholang_ast::SyncSendCont::Empty { line_num, col_num } => {
                Box::new(rholang_ast::Block {
                    proc: Proc::Nil {
                        line_num: *line_num,
                        col_num: *col_num,
                    },
                    line_num: *line_num,
                    col_num: *col_num,
                })
            }

            rholang_ast::SyncSendCont::NonEmpty {
                proc,
                line_num,
                col_num,
            } => Box::new(rholang_ast::Block {
                proc: *(proc).clone(),
                line_num: *line_num,
                col_num: *col_num,
            }),
        };

        Proc::Input {
            formals: list_receipt,
            proc,
            line_num,
            col_num,
        }
    };

    let list_name: Vec<NameDecl> = vec![NameDecl {
        var: rholang_ast::Var {
            name: identifier,
            line_num,
            col_num,
        },
        uri: None,
        line_num,
        col_num,
    }];

    let decls: rholang_ast::Decls = rholang_ast::Decls {
        decls: list_name,
        line_num,
        col_num,
    };

    let p_par = Proc::Par {
        left: Box::new(send),
        right: Box::new(receive),
        line_num,
        col_num,
    };

    let p_new: Proc = Proc::New {
        decls,
        proc: Box::new(p_par),
        line_num,
        col_num,
    };

    normalize_match_proc(&p_new, input, env)
}

/// Parallel version of normalize_p_send_sync for new AST SendSync
pub fn normalize_p_send_sync_new_ast<'ast>(
    channel: &'ast AnnName<'ast>,
    messages: &'ast rholang_parser::ast::ProcList<'ast>,
    cont: &rholang_parser::ast::SyncSendCont<'ast>,
    span: &rholang_parser::SourceSpan,
    input: ProcVisitInputsSpan,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputsSpan, InterpreterError> {
    let identifier = Uuid::new_v4().to_string();

    // Create variable name for the response channel
    // TODO: Replace Box::leak with proper arena allocation for Name
    let name_var = Box::leak(Box::new(rholang_parser::ast::Name::ProcVar(
        rholang_parser::ast::Var::Id(rholang_parser::ast::Id {
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
            proc: parser
                .ast_builder()
                .alloc_eval(rholang_parser::ast::AnnName {
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
            proc: parser.ast_builder().alloc_send(
                rholang_parser::ast::SendType::Single,
                *channel,
                &listproc,
            ),
            span: *span,
        }
    };

    // Build the receive process: for (_ <- name_var) { cont }
    let receive: AnnProc = {
        // Create wildcard pattern
        let wildcard = rholang_parser::ast::AnnName {
            name: rholang_parser::ast::Name::ProcVar(rholang_parser::ast::Var::Wildcard),
            span: *span,
        };

        // Create bind for the pattern: _ <- name_var
        let bind = rholang_parser::ast::Bind::Linear {
            lhs: rholang_parser::ast::Names {
                names: smallvec::SmallVec::from_vec(vec![wildcard]),
                remainder: None,
            },
            rhs: rholang_parser::ast::Source::Simple {
                name: rholang_parser::ast::AnnName {
                    name: *name_var,
                    span: *span,
                },
            },
        };

        // Create receipt containing the bind
        let receipt: smallvec::SmallVec<[rholang_parser::ast::Bind<'ast>; 1]> =
            smallvec::SmallVec::from_vec(vec![bind]);
        let receipts: smallvec::SmallVec<
            [smallvec::SmallVec<[rholang_parser::ast::Bind<'ast>; 1]>; 1],
        > = smallvec::SmallVec::from_vec(vec![receipt]);

        // Get the continuation process
        let cont_proc = match cont {
            rholang_parser::ast::SyncSendCont::Empty => AnnProc {
                proc: parser.ast_builder().const_nil(),
                span: *span,
            },
            rholang_parser::ast::SyncSendCont::NonEmpty(proc) => *proc,
        };

        AnnProc {
            proc: parser.ast_builder().alloc_for(receipts, cont_proc),
            span: *span,
        }
    };

    // Create name declaration for the new variable
    let name_decl = rholang_parser::ast::NameDecl {
        id: rholang_parser::ast::Id {
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

    // Normalize the constructed AST
    normalize_ann_proc(&p_new, input, env, parser)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::rust::interpreter::compiler::bound_map_chain::BoundMapChain;
    use crate::rust::interpreter::compiler::exports::{BoundMapChainSpan, FreeMap, FreeMapSpan};
    use crate::rust::interpreter::compiler::normalize::{ProcVisitInputs, VarSort};
    use crate::rust::interpreter::compiler::rholang_ast;
    use crate::rust::interpreter::compiler::rholang_ast::Proc;
    use models::rhoapi::Par;

    fn p_send_sync() -> Proc {
        let p_send_sync = Proc::SendSync {
            name: rholang_ast::Name::ProcVar(Box::new(rholang_ast::Proc::Wildcard {
                line_num: 0,
                col_num: 0,
            })),
            messages: rholang_ast::ProcList {
                procs: vec![],
                line_num: 1,
                col_num: 1,
            },
            cont: rholang_ast::SyncSendCont::Empty {
                line_num: 2,
                col_num: 2,
            },
            line_num: 3,
            col_num: 3,
        };

        p_send_sync
    }

    #[test]
    fn test_normalize_p_send_sync() {
        let p = p_send_sync();
        fn inputs() -> ProcVisitInputs {
            ProcVisitInputs {
                par: Par::default(),
                bound_map_chain: BoundMapChain::new(),
                free_map: FreeMap::<VarSort>::new(),
            }
        }

        let env = HashMap::<String, Par>::new();

        let result = match p {
            Proc::SendSync {
                name,
                messages,
                cont,
                line_num,
                col_num,
            } => normalize_p_send_sync(&name, &messages, &cont, line_num, col_num, inputs(), &env),
            _ => Result::Err(InterpreterError::NormalizerError(
                "Expected Proc::SendSync".to_string(),
            )),
        };

        assert!(result.is_ok());

        // check the result
        // let result = result.unwrap();
        // let par = result.par;
        // assert_eq!(par.sends.len(), 1);
        // assert_eq!(par.receives.len(), 1);
    }

    #[test]
    fn new_ast_test_normalize_p_send_sync() {
        // Maps to original: test_normalize_p_send_sync
        use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
        use rholang_parser::ast::{
            AnnName, AnnProc, Name as NewName, Proc as NewProc, SyncSendCont as NewSyncSendCont,
            Var as NewVar,
        };
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

        // Create wildcard << []; Nil using new AST (same as original test)
        let send_sync_proc = AnnProc {
            proc: Box::leak(Box::new(NewProc::SendSync {
                channel: AnnName {
                    name: NewName::ProcVar(NewVar::Wildcard),
                    span: SourceSpan {
                        start: SourcePos { line: 0, col: 0 },
                        end: SourcePos { line: 0, col: 0 },
                    },
                },
                messages: smallvec::SmallVec::new(), // Empty messages
                cont: NewSyncSendCont::Empty,
            })),
            span: SourceSpan {
                start: SourcePos { line: 3, col: 3 },
                end: SourcePos { line: 3, col: 3 },
            },
        };

        let result = normalize_ann_proc(&send_sync_proc, inputs(), &env, &parser);
        assert!(result.is_ok());

        // The result should be normalized successfully
        // Note: The original test doesn't check specific structure, just that it succeeds
    }
}
