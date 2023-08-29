.. title:: clang-tidy - readability-function-size

readability-function-size
=========================

`google-readability-function-size` redirects here as an alias for this check.

Checks for large functions based on various metrics.

Options
-------

.. option:: LineThreshold

   Flag functions exceeding this number of lines. This option is disabled by 
   default.

.. option:: StatementThreshold

   Flag functions exceeding this number of statements. This may differ
   significantly from the number of lines for macro-heavy code. The default is
   `800`.

.. option:: BranchThreshold

   Flag functions exceeding this number of control statements. This option is 
   disabled by default.

.. option:: ParameterThreshold

   Flag functions that exceed a specified number of parameters. This option 
   is disabled by default.

.. option:: NestingThreshold

    Flag compound statements which create next nesting level after
    `NestingThreshold`. This may differ significantly from the expected value
    for macro-heavy code. This option is disabled by default.

.. option:: VariableThreshold

   Flag functions exceeding this number of variables declared in the body.
   Please note that function parameters and variables declared in lambdas,
   GNU Statement Expressions, and nested class inline functions are not 
   counted. This option is disabled by default.
