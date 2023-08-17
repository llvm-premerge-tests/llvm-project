// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.cplusplus.ArrayDelete -std=c++11 -verify -analyzer-output=text %s

struct Base {
    virtual ~Base() = default;
};

struct Derived : public Base {};

struct DoubleDerived : public Derived {};

Derived *get();

Base *create() {
    Base *b = new Derived[3]; // expected-note{{Conversion from derived to base happened here}}
    return b;
}

void sink(Base *b) {
    delete[] b; // expected-warning{{Deleting an array of polymorphic objects is undefined}}
    // expected-note@-1{{Deleting an array of polymorphic objects is undefined}}
}

void sink_cast(Base *b) {
    delete[] reinterpret_cast<Derived*>(b); // no-warning
}

void sink_derived(Derived *d) {
    delete[] d; // no-warning
}

void same_function() {
    Base *sd = new Derived[10]; // expected-note{{Conversion from derived to base happened here}}
    delete[] sd; // expected-warning{{Deleting an array of polymorphic objects is undefined}}
    // expected-note@-1{{Deleting an array of polymorphic objects is undefined}}
    
    Base *dd = new DoubleDerived[10]; // expected-note{{Conversion from derived to base happened here}}
    delete[] dd; // expected-warning{{Deleting an array of polymorphic objects is undefined}}
    // expected-note@-1{{Deleting an array of polymorphic objects is undefined}}
}

void different_function() {
    Base *assigned = get(); // expected-note{{Conversion from derived to base happened here}}
    delete[] assigned; // expected-warning{{Deleting an array of polymorphic objects is undefined}}
    // expected-note@-1{{Deleting an array of polymorphic objects is undefined}}

    Base *indirect;
    indirect = get(); // expected-note{{Conversion from derived to base happened here}}
    delete[] indirect; // expected-warning{{Deleting an array of polymorphic objects is undefined}}
    // expected-note@-1{{Deleting an array of polymorphic objects is undefined}}

    Base *created = create(); // expected-note{{Calling 'create'}}
    // expected-note@-1{{Returning from 'create'}}
    delete[] created; // expected-warning{{Deleting an array of polymorphic objects is undefined}}
    // expected-note@-1{{Deleting an array of polymorphic objects is undefined}}

    Base *sb = new Derived[10]; // expected-note{{Conversion from derived to base happened here}}
    sink(sb); // expected-note{{Calling 'sink'}}
}

void safe_function() {
    Derived *d = new Derived[10];
    delete[] d; // no-warning

    Base *b = new Derived[10];
    delete[] reinterpret_cast<Derived*>(b); // no-warning

    Base *sb = new Derived[10];
    sink_cast(sb); // no-warning

    Derived *sd = new Derived[10];
    sink_derived(sd); // no-warning
}
