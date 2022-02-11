#[macro_export]
macro_rules! zip_apply {
    (@build $pat:pat in $v:expr; ; $body:stmt) => {
        for $pat in $v {
            $body
        }
    };
    (@build $pat:pat in $v:expr; $first_pat:pat in $first_v:expr $(,$rest_pat:pat in $rest_v:expr)* $(,)? ; $body:stmt) => {
        $crate::zip_apply!(@build ($pat, $first_pat) in $v.zip($first_v); $($rest_pat in $rest_v,)* ; $body)
    };
    (@build $first_pat:pat in $first_v:expr $(,$rest_pat:pat in $rest_v:expr)* $(,)? ; $body:stmt) => {
        $crate::zip_apply!(@build $first_pat in $first_v; $($rest_pat in $rest_v,)* ; $body)
    };
    (@build $($t:tt)*) => { compile_error!("Invalid syntax in apply!()") };
    ($($t:tt)*) => {
        $crate::zip_apply!(@build $($t)*)
    };
}
