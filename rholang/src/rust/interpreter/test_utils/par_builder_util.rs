use crate::rust::interpreter::compiler::compiler::Compiler;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use rholang_parser::ast::{
    AnnName as NewAnnName, AnnProc as NewAnnProc, BinaryExpOp, Id, KeyValuePair as NewKeyValuePair,
    Name as NewName, Names as NewNames, Var as NewVar,
};
use rholang_parser::{SourcePos, SourceSpan};
use std::collections::HashMap;

pub struct ParBuilderUtil;

// TODO: Review source spans

impl ParBuilderUtil {
    pub fn mk_term(rho: &str) -> Result<Par, InterpreterError> {
        Compiler::source_to_adt_with_normalizer_env(rho, HashMap::new())
    }

    pub fn assert_compiled_equal(s: &str, t: &str) {
        let par_s = ParBuilderUtil::mk_term(s).expect("Compilation failed for the first string");
        let par_t = ParBuilderUtil::mk_term(t).expect("Compilation failed for the second string");
        assert_eq!(par_s, par_t, "Compiled Par values are not equal");
    }

    pub fn new_ast_proc_var<'ast>(
        name: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: parser.ast_builder().alloc_var(Id {
                name,
                pos: SourcePos { line: 0, col: 0 },
            }),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn new_ast_eval_name_var<'ast>(
        name: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: parser.ast_builder().alloc_eval(NewAnnName {
                name: NewName::ProcVar(NewVar::Id(Id {
                    name,
                    pos: SourcePos { line: 0, col: 0 },
                })),
                span: SourceSpan {
                    start: SourcePos { line: 1, col: 1 },
                    end: SourcePos { line: 1, col: 1 },
                },
            }),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn new_ast_int<'ast>(
        value: i64,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: parser.ast_builder().alloc_long_literal(value),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn new_ast_string<'ast>(
        value: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: parser.ast_builder().alloc_string_literal(value),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn new_ast_par<'ast>(
        left: NewAnnProc<'ast>,
        right: NewAnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: parser.ast_builder().alloc_par(left, right),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn new_ast_add<'ast>(
        left: NewAnnProc<'ast>,
        right: NewAnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: parser
                .ast_builder()
                .alloc_binary_exp(BinaryExpOp::Add, left, right),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating "P + R" (add with par of var)
    pub fn new_ast_add_with_par_of_var<'ast>(
        var1: &'ast str,
        var2: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        Self::new_ast_add(
            Self::new_ast_proc_var(var1, parser),
            Self::new_ast_proc_var(var2, parser),
            parser,
        )
    }

    // Helper for creating "8 | Q" (par with int and var)
    pub fn new_ast_par_with_int_and_var<'ast>(
        int_val: i64,
        var_name: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        Self::new_ast_par(
            Self::new_ast_int(int_val, parser),
            Self::new_ast_proc_var(var_name, parser),
            parser,
        )
    }

    // Helper for creating key-value pairs for maps
    pub fn new_ast_key_value_pair<'ast>(
        key: NewAnnProc<'ast>,
        value: NewAnnProc<'ast>,
    ) -> NewKeyValuePair<'ast> {
        (key, value)
    }

    // Helper for creating collections
    pub fn new_ast_list<'ast>(
        elements: Vec<NewAnnProc<'ast>>,
        remainder: Option<NewVar<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: match remainder {
                Some(r) => parser.ast_builder().alloc_list_with_remainder(&elements, r),
                None => parser.ast_builder().alloc_list(&elements),
            },
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn new_ast_set<'ast>(
        elements: Vec<NewAnnProc<'ast>>,
        remainder: Option<NewVar<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: match remainder {
                Some(r) => parser.ast_builder().alloc_set_with_remainder(&elements, r),
                None => parser.ast_builder().alloc_set(&elements),
            },
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn new_ast_map<'ast>(
        pairs: Vec<NewKeyValuePair<'ast>>,
        remainder: Option<NewVar<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        // Flatten key-value pairs into a flat array for the arena method
        let flat_pairs: Vec<NewAnnProc<'ast>> =
            pairs.into_iter().flat_map(|(k, v)| vec![k, v]).collect();
        NewAnnProc {
            proc: match remainder {
                Some(r) => parser
                    .ast_builder()
                    .alloc_map_with_remainder(&flat_pairs, r),
                None => parser.ast_builder().alloc_map(&flat_pairs),
            },
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating variables for remainder
    pub fn new_ast_var<'ast>(name: &'ast str) -> NewVar<'ast> {
        NewVar::Id(Id {
            name,
            pos: SourcePos { line: 0, col: 0 },
        })
    }

    // Helper for creating AnnName
    pub fn new_ast_ann_name<'ast>(name: NewName<'ast>) -> NewAnnName<'ast> {
        NewAnnName {
            name,
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating AnnName from variable name
    pub fn new_ast_ann_name_from_var<'ast>(name: &'ast str) -> NewAnnName<'ast> {
        Self::new_ast_ann_name(NewName::ProcVar(NewVar::Id(Id {
            name,
            pos: SourcePos { line: 0, col: 0 },
        })))
    }

    // Helper for creating Names structure
    pub fn new_ast_names<'ast>(
        names: Vec<NewAnnName<'ast>>,
        remainder: Option<NewVar<'ast>>,
    ) -> NewNames<'ast> {
        use smallvec::SmallVec;
        NewNames {
            names: SmallVec::from_vec(names),
            remainder,
        }
    }

    // Helper for creating Contract
    pub fn new_ast_contract<'ast>(
        name: NewAnnName<'ast>,
        formals: NewNames<'ast>,
        body: NewAnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        NewAnnProc {
            proc: parser.ast_builder().alloc_contract(name, formals, body),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating New declaration
    pub fn new_ast_new<'ast>(
        decls: Vec<NewVar<'ast>>,
        proc: NewAnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> NewAnnProc<'ast> {
        use rholang_parser::ast::NameDecl;
        let name_decls: Vec<NameDecl<'ast>> = decls
            .into_iter()
            .map(|var| match var {
                NewVar::Id(id) => NameDecl { id, uri: None },
                NewVar::Wildcard => NameDecl {
                    id: Id {
                        name: "_",
                        pos: SourcePos { line: 0, col: 0 },
                    },
                    uri: None,
                },
            })
            .collect();

        NewAnnProc {
            proc: parser.ast_builder().alloc_new(proc, name_decls),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }
}
