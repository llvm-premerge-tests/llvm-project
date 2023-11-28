// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -verify %s

int *glob;
static int *static_glob;

auto lambda_global = []() {
  ++glob; // expected-warning{{unsafe arithmetic over raw pointer global variable 'glob'}}
          // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  ++static_glob; // expected-warning{{unsafe arithmetic over raw pointer global variable 'static_glob'}}
                 // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
};

class C {
  int *memb;
  static int *static_memb;

public:
  void foo() {
    int *loc;
    ++loc; // expected-warning{{unsafe arithmetic over raw pointer local variable 'loc'}}
           // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

    static int *static_loc;
    ++static_loc; // expected-warning{{unsafe arithmetic over raw pointer static local variable 'static_loc'}}
                  // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

    [=]() {
      loc[5] = 10; // expected-warning{{unsafe buffer access through captured raw pointer local variable 'loc'}}
                   // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
      static_loc[5] = 10; // expected-warning{{unsafe buffer access through raw pointer static local variable 'static_loc'}}
                          // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
    };
    [&]() {
      ++loc; // expected-warning{{unsafe arithmetic over captured raw pointer local variable 'loc'}}
             // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
      ++static_loc; // expected-warning{{unsafe arithmetic over raw pointer static local variable 'static_loc'}}
                    // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
    };

    ++memb; // expected-warning{{unsafe arithmetic over raw pointer member variable 'memb'}}
            // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
    ++static_memb; // expected-warning{{unsafe arithmetic over raw pointer static member variable 'static_memb'}}
                   // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

    // Interestingly, explicit 'this->' produces a completely different AST.
    ++this->static_memb; // expected-warning{{unsafe arithmetic over raw pointer static member variable 'static_memb'}}
                         // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

    [this]() {
      ++memb; // expected-warning{{unsafe arithmetic over raw pointer member variable 'memb'}}
              // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
      ++static_memb; // expected-warning{{unsafe arithmetic over raw pointer static member variable 'static_memb'}}
                     // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
    ++this->static_memb; // expected-warning{{unsafe arithmetic over raw pointer static member variable 'static_memb'}}
                         // expected-note@-1{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
    };
  }
};
