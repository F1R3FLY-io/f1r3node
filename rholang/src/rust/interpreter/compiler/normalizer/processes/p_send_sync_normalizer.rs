use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::normalize::normalize_ann_proc;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use std::collections::HashMap;
use uuid::Uuid;

use rholang_parser::ast::{AnnProc, Bind, Id, Name, SendType, SyncSendCont};

pub fn normalize_p_send_sync<'ast>(
    channel: &'ast Name<'ast>,
    messages: &'ast rholang_parser::ast::ProcList<'ast>,
    cont: &SyncSendCont<'ast>,
    span: &rholang_parser::SourceSpan,
    input: ProcVisitInputs,
    env: &HashMap<String, Par>,
    parser: &'ast rholang_parser::RholangParser<'ast>,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let identifier = Uuid::new_v4().to_string();

    // Allocate identifier string in the parser's string arena
    let identifier_str = parser.ast_builder().alloc_str(&identifier);

    // Create variable name for the response channel
    let name_var = rholang_parser::ast::Name::NameVar(rholang_parser::ast::Var::Id(Id {
        name: identifier_str,
        pos: span.start,
    }));

    // Build the send process: channel!(name_var, ...messages)
    let send: AnnProc = {
        let mut listproc = Vec::new();

        // Add the response channel name as first argument
        listproc.push(AnnProc {
            proc: parser.ast_builder().alloc_eval(name_var),
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
        let wildcard = rholang_parser::ast::Name::NameVar(rholang_parser::ast::Var::Wildcard);

        // Create bind for the pattern: _ <- name_var
        let bind = Bind::Linear {
            lhs: rholang_parser::ast::Names {
                names: smallvec::SmallVec::from_vec(vec![wildcard]),
                remainder: None,
            },
            rhs: rholang_parser::ast::Source::Simple { name: name_var },
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
            name: identifier_str,
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
    use crate::rust::interpreter::compiler::exports::{BoundMapChain, FreeMap};
    use crate::rust::interpreter::compiler::normalize::VarSort;
    use models::rhoapi::Par;

    #[test]
    fn p_send_sync_should_normalize_a_basic_send_sync() {
        use rholang_parser::ast::{Name, Var};
        use rholang_parser::{SourcePos, SourceSpan};

        fn inputs() -> ProcVisitInputs {
            ProcVisitInputs {
                par: Par::default(),
                bound_map_chain: BoundMapChain::new(),
                free_map: FreeMap::<VarSort>::new(),
            }
        }

        let env = HashMap::<String, Par>::new();
        let parser = rholang_parser::RholangParser::new();

        let channel = Name::NameVar(Var::Wildcard);

        let messages = smallvec::SmallVec::new();

        let cont = rholang_parser::ast::SyncSendCont::Empty;

        let span = SourceSpan {
            start: SourcePos { line: 3, col: 3 },
            end: SourcePos { line: 3, col: 3 },
        };

        let result =
            normalize_p_send_sync(&channel, &messages, &cont, &span, inputs(), &env, &parser);
        assert!(result.is_ok());
    }
}
