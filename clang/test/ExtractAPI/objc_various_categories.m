// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api -x objective-c-header \
// RUN: -target arm64-apple-macosx \
// RUN: %t/myclass_1.h \
// RUN: %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
#import "myclass_1.h"
#import "myclass_2.h"
#import "Foundation.h"

@interface MyClass1 (MyCategory1)
- (int) SomeMethod;
@end

@interface MyClass2 (MyCategory2)
- (int) SomeMethod2;
@end

@interface NSString (Category1)
-(void) StringMethod;
@end

@interface NSString (Category2)
-(void) StringMethod2;
@end

//--- myclass_1.h
@interface MyClass1
@end

//--- myclass_2.h
@interface MyClass2
@end

//--- Foundation.h
@interface NSString
@end

//--- reference.output.json.in
{
  "metadata": {
    "formatVersion": {
      "major": 0,
      "minor": 5,
      "patch": 3
    },
    "generator": "?"
  },
  "module": {
    "name": "",
    "platform": {
      "architecture": "arm64",
      "operatingSystem": {
        "minimumVersion": {
          "major": 11,
          "minor": 0,
          "patch": 0
        },
        "name": "macosx"
      },
      "vendor": "apple"
    }
  },
  "relationships": [
    {
      "kind": "memberOf",
      "source": "c:objc(cs)MyClass1(im)SomeMethod",
      "target": "c:objc(cs)MyClass1",
      "targetFallback": "MyClass1"
    },
    {
      "kind": "extensionTo",
      "source": "c:objc(cy)MyClass2@MyCategory2",
      "target": "c:objc(cs)MyClass2",
      "targetFallback": "MyClass2"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)MyClass2(im)SomeMethod2",
      "target": "c:objc(cy)MyClass2@MyCategory2",
      "targetFallback": "MyCategory2"
    },
    {
      "kind": "extensionTo",
      "source": "c:objc(cy)NSString@Category1",
      "target": "c:objc(cs)NSString",
      "targetFallback": "NSString"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)NSString(im)StringMethod",
      "target": "c:objc(cy)NSString@Category1",
      "targetFallback": "Category1"
    },
    {
      "kind": "extensionTo",
      "source": "c:objc(cy)NSString@Category2",
      "target": "c:objc(cs)NSString",
      "targetFallback": "NSString"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)NSString(im)StringMethod2",
      "target": "c:objc(cy)NSString@Category2",
      "targetFallback": "Category2"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@interface"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyClass1"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)MyClass1"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "objective-c.class"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 1
        },
        "uri": "file://INPUT_DIR/myclass_1.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyClass1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyClass1"
          }
        ],
        "title": "MyClass1"
      },
      "pathComponents": [
        "MyClass1"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "SomeMethod"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:I",
            "spelling": "int"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)MyClass1(im)SomeMethod"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "SomeMethod"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "SomeMethod"
          }
        ],
        "title": "SomeMethod"
      },
      "pathComponents": [
        "MyClass1",
        "SomeMethod"
      ]
    },
    {
      "accessLevel": "public",
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)MyClass2@MyCategory2"
      },
      "kind": {
        "displayName": "Module Extension",
        "identifier": "objective-c.module.extension"
      }
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@interface"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:objc(cs)MyClass2",
          "spelling": "MyClass2"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "identifier",
          "spelling": "MyCategory2"
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)MyClass2@MyCategory2"
      },
      "kind": {
        "displayName": "Class Extension",
        "identifier": "objective-c.class.extension"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyCategory2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyCategory2"
          }
        ],
        "title": "MyClass2 (MyCategory2)"
      },
      "pathComponents": [
        "MyCategory2"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "SomeMethod2"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:I",
            "spelling": "int"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)MyClass2(im)SomeMethod2"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 10
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "SomeMethod2"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "SomeMethod2"
          }
        ],
        "title": "SomeMethod2"
      },
      "pathComponents": [
        "MyCategory2",
        "SomeMethod2"
      ]
    },
    {
      "accessLevel": "public",
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)NSString@Category1"
      },
      "kind": {
        "displayName": "Module Extension",
        "identifier": "objective-c.module.extension"
      }
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@interface"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:objc(cs)NSString",
          "spelling": "NSString"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "identifier",
          "spelling": "Category1"
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)NSString@Category1"
      },
      "kind": {
        "displayName": "Class Extension",
        "identifier": "objective-c.class.extension"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 13
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Category1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Category1"
          }
        ],
        "title": "NSString (Category1)"
      },
      "pathComponents": [
        "Category1"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "StringMethod"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)NSString(im)StringMethod"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 14
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "StringMethod"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "StringMethod"
          }
        ],
        "title": "StringMethod"
      },
      "pathComponents": [
        "Category1",
        "StringMethod"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@interface"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:objc(cs)NSString",
          "spelling": "NSString"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "identifier",
          "spelling": "Category2"
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)NSString@Category2"
      },
      "kind": {
        "displayName": "Class Extension",
        "identifier": "objective-c.class.extension"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 17
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Category2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Category2"
          }
        ],
        "title": "NSString (Category2)"
      },
      "pathComponents": [
        "Category2"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "StringMethod2"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)NSString(im)StringMethod2"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 18
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "StringMethod2"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "StringMethod2"
          }
        ],
        "title": "StringMethod2"
      },
      "pathComponents": [
        "Category2",
        "StringMethod2"
      ]
    }
  ]
}
