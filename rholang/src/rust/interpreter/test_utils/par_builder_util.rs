use crate::rust::interpreter::compiler::compiler::Compiler;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use rholang_parser::ast::{AnnProc, BinaryExpOp, Id, KeyValuePair, Name, Names, Var};
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

    pub fn create_ast_proc_var<'ast>(
        name: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
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

    pub fn create_ast_eval_name_var<'ast>(
        name: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_eval(Name::NameVar(Var::Id(Id {
                name,
                pos: SourcePos { line: 0, col: 0 },
            }))),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn create_ast_int<'ast>(
        value: i64,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_long_literal(value),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn create_ast_string<'ast>(
        value: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_string_literal(value),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn create_ast_par<'ast>(
        left: AnnProc<'ast>,
        right: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_par(left, right),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    pub fn create_ast_add<'ast>(
        left: AnnProc<'ast>,
        right: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
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
    pub fn create_ast_add_with_par_of_var<'ast>(
        var1: &'ast str,
        var2: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        Self::create_ast_add(
            Self::create_ast_proc_var(var1, parser),
            Self::create_ast_proc_var(var2, parser),
            parser,
        )
    }

    // Helper for creating "8 | Q" (par with int and var)
    pub fn create_ast_par_with_int_and_var<'ast>(
        int_val: i64,
        var_name: &'ast str,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        Self::create_ast_par(
            Self::create_ast_int(int_val, parser),
            Self::create_ast_proc_var(var_name, parser),
            parser,
        )
    }

    // Helper for creating key-value pairs for maps
    pub fn create_ast_key_value_pair<'ast>(
        key: AnnProc<'ast>,
        value: AnnProc<'ast>,
    ) -> KeyValuePair<'ast> {
        (key, value)
    }

    // Helper for creating collections
    pub fn create_ast_list<'ast>(
        elements: Vec<AnnProc<'ast>>,
        remainder: Option<Var<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
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

    pub fn create_ast_set<'ast>(
        elements: Vec<AnnProc<'ast>>,
        remainder: Option<Var<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
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

    pub fn create_ast_map<'ast>(
        pairs: Vec<KeyValuePair<'ast>>,
        remainder: Option<Var<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        // Flatten key-value pairs into a flat array for the arena method
        let flat_pairs: Vec<AnnProc<'ast>> =
            pairs.into_iter().flat_map(|(k, v)| vec![k, v]).collect();
        AnnProc {
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
    pub fn create_ast_var<'ast>(name: &'ast str) -> Var<'ast> {
        Var::Id(Id {
            name,
            pos: SourcePos { line: 0, col: 0 },
        })
    }

    // Helper for creating Name from variable name
    pub fn create_ast_name_from_var<'ast>(name: &'ast str) -> Name<'ast> {
        Name::NameVar(Var::Id(Id {
            name,
            pos: SourcePos { line: 0, col: 0 },
        }))
    }

    // Helper for creating Names structure
    pub fn create_ast_names<'ast>(
        names: Vec<Name<'ast>>,
        remainder: Option<Var<'ast>>,
    ) -> Names<'ast> {
        use smallvec::SmallVec;
        Names {
            names: SmallVec::from_vec(names),
            remainder,
        }
    }

    // Helper for creating Contract
    pub fn create_ast_contract<'ast>(
        name: Name<'ast>,
        formals: Names<'ast>,
        body: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_contract(name, formals, body),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating New declaration
    pub fn create_ast_new<'ast>(
        decls: Vec<Var<'ast>>,
        proc: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        use rholang_parser::ast::NameDecl;
        let name_decls: Vec<NameDecl<'ast>> = decls
            .into_iter()
            .map(|var| match var {
                Var::Id(id) => NameDecl { id, uri: None },
                Var::Wildcard => NameDecl {
                    id: Id {
                        name: "_",
                        pos: SourcePos { line: 0, col: 0 },
                    },
                    uri: None,
                },
            })
            .collect();

        AnnProc {
            proc: parser.ast_builder().alloc_new(proc, name_decls),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }
}
