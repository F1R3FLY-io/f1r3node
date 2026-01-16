use crate::rust::interpreter::compiler::compiler::Compiler;
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Par;
use rholang_parser::ast::{
    AnnProc, BinaryExpOp, BundleType, Id, KeyValuePair, Name, Names, SendType, SimpleType, Var,
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
                Var::Id(id) => NameDecl { id, space_type: None, uri: None },
                Var::Wildcard => NameDecl {
                    id: Id {
                        name: "_",
                        pos: SourcePos { line: 0, col: 0 },
                    },
                    space_type: None,
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

    // Helper for creating Bundle
    pub fn create_ast_bundle<'ast>(
        bundle_type: BundleType,
        proc: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_bundle(bundle_type, proc),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Send
    pub fn create_ast_send<'ast>(
        channel: Name<'ast>,
        send_type: SendType,
        inputs: Vec<AnnProc<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_send(send_type, channel, None, &inputs),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Nil
    pub fn create_ast_nil<'ast>(
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().const_nil(),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating SimpleType
    pub fn create_ast_simple_type<'ast>(
        simple_type: SimpleType,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_simple_type(simple_type),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Name::Quote wrapping a proc
    pub fn create_ast_quote_name<'ast>(proc: AnnProc<'ast>) -> Name<'ast> {
        Name::Quote(proc)
    }

    // Helper for creating Wildcard proc
    pub fn create_ast_wildcard<'ast>(
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().const_wild(),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating boolean literal
    pub fn create_ast_bool_literal<'ast>(
        value: bool,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: if value {
                parser.ast_builder().const_true()
            } else {
                parser.ast_builder().const_false()
            },
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating long literal
    pub fn create_ast_long_literal<'ast>(
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

    // Helper for creating binary expression
    pub fn create_ast_binary_exp<'ast>(
        op: rholang_parser::ast::BinaryExpOp,
        left: AnnProc<'ast>,
        right: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_binary_exp(op, left, right),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating name declaration from string
    pub fn create_ast_name_decl_from_str<'ast>(
        name: &'ast str,
    ) -> rholang_parser::ast::NameDecl<'ast> {
        rholang_parser::ast::NameDecl {
            id: rholang_parser::ast::Id {
                name,
                pos: SourcePos { line: 0, col: 0 },
            },
            space_type: None,
            uri: None,
        }
    }

    // Helper for creating if-then-else
    pub fn create_ast_if_then_else<'ast>(
        condition: AnnProc<'ast>,
        if_true: AnnProc<'ast>,
        if_false: Option<AnnProc<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser
                .ast_builder()
                .alloc_if_then_else_opt(condition, if_true, if_false),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Eval
    pub fn create_ast_eval<'ast>(
        name: rholang_parser::ast::Name<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_eval(name),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating empty list
    pub fn create_ast_empty_list<'ast>(
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_list(&[]),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating list with remainder
    pub fn create_ast_list_remainder<'ast>(
        elements: Vec<AnnProc<'ast>>,
        remainder: rholang_parser::ast::Var<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser
                .ast_builder()
                .alloc_list_with_remainder(&elements, remainder),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating ForComprehension
    pub fn create_ast_for_comprehension<'ast>(
        receipts: Vec<Vec<rholang_parser::ast::Bind<'ast>>>,
        body: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_for(receipts, body),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Name::NameVar from string
    pub fn create_ast_name_var<'ast>(name: &'ast str) -> rholang_parser::ast::Name<'ast> {
        use rholang_parser::ast::{Id, Name, Var};
        Name::NameVar(Var::Id(Id {
            name,
            pos: SourcePos { line: 0, col: 0 },
        }))
    }

    // Helper for creating string literal
    pub fn create_ast_string_literal<'ast>(
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

    // Helper for creating Match
    pub fn create_ast_match<'ast>(
        expression: AnnProc<'ast>,
        cases: Vec<rholang_parser::ast::Case<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        // Convert cases to flat pattern/proc pairs for alloc_match
        let mut flat_cases = Vec::new();
        for case in cases {
            flat_cases.push(case.pattern);
            flat_cases.push(case.proc);
        }
        AnnProc {
            proc: parser.ast_builder().alloc_match(expression, &flat_cases),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Proc::ProcVar
    pub fn create_ast_proc_var_from_var<'ast>(
        var: rholang_parser::ast::Var<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_proc_var(var),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating UnaryExp
    pub fn create_ast_unary_exp<'ast>(
        op: rholang_parser::ast::UnaryExpOp,
        arg: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_unary_exp(op, arg),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Method
    pub fn create_ast_method<'ast>(
        name: rholang_parser::ast::Id<'ast>,
        receiver: AnnProc<'ast>,
        args: Vec<AnnProc<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_method(name, receiver, &args),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating FunctionCall (built-in functions like getSpaceAgent)
    pub fn create_ast_function_call<'ast>(
        name: rholang_parser::ast::Id<'ast>,
        args: Vec<AnnProc<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_function_call(name, &args),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating New with NameDecl
    pub fn create_ast_new_with_decls<'ast>(
        decls: Vec<rholang_parser::ast::NameDecl<'ast>>,
        proc: AnnProc<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_new(proc, decls),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating VarRef
    pub fn create_ast_var_ref<'ast>(
        kind: rholang_parser::ast::VarRefKind,
        var: rholang_parser::ast::Id<'ast>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_var_ref(kind, var),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }

    // Helper for creating Tuple
    pub fn create_ast_tuple<'ast>(
        elements: Vec<AnnProc<'ast>>,
        parser: &'ast rholang_parser::RholangParser<'ast>,
    ) -> AnnProc<'ast> {
        AnnProc {
            proc: parser.ast_builder().alloc_tuple(&elements),
            span: SourceSpan {
                start: SourcePos { line: 0, col: 0 },
                end: SourcePos { line: 0, col: 0 },
            },
        }
    }
}
