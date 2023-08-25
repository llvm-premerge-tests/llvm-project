; REQUIRES: x86

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.Native = type { %struct.A }

; Placeholders for testing
@_ZTVN10__cxxabiv117__class_type_infoE = linkonce_odr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr null] }
@_ZTVN10__cxxabiv120__si_class_type_infoE = linkonce_odr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr null] }

@_ZTV6Native = linkonce_odr unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI6Native, ptr @_ZN1A1nEi, ptr @_ZN6Native1fEi] }
@_ZTS6Native = linkonce_odr constant [8 x i8] c"6Native\00"
@_ZTI6Native = linkonce_odr constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS6Native, ptr @_ZTI1A }

; Base type A does not need to emit a vtable if it's never instantiated. However, RTTI still gets generated
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"
@_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }


define linkonce_odr i32 @_ZN6Native1fEi(ptr %this, i32 %a) #0 {
   ret i32 1;
}

define linkonce_odr i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }
