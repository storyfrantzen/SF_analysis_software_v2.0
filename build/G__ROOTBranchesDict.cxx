// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME G__ROOTBranchesDict
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "ROOT/RConfig.hxx"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Header files passed as explicit arguments
#include "/work/clas12/storyf/SF_analysis_software_v2.0/include/ROOTBranches.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ROOT {
   static void *new_EventBranches(void *p = nullptr);
   static void *newArray_EventBranches(Long_t size, void *p);
   static void delete_EventBranches(void *p);
   static void deleteArray_EventBranches(void *p);
   static void destruct_EventBranches(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::EventBranches*)
   {
      ::EventBranches *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::EventBranches >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("EventBranches", ::EventBranches::Class_Version(), "ROOTBranches.h", 25,
                  typeid(::EventBranches), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::EventBranches::Dictionary, isa_proxy, 4,
                  sizeof(::EventBranches) );
      instance.SetNew(&new_EventBranches);
      instance.SetNewArray(&newArray_EventBranches);
      instance.SetDelete(&delete_EventBranches);
      instance.SetDeleteArray(&deleteArray_EventBranches);
      instance.SetDestructor(&destruct_EventBranches);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::EventBranches*)
   {
      return GenerateInitInstanceLocal(static_cast<::EventBranches*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::EventBranches*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_RecBranches(void *p = nullptr);
   static void *newArray_RecBranches(Long_t size, void *p);
   static void delete_RecBranches(void *p);
   static void deleteArray_RecBranches(void *p);
   static void destruct_RecBranches(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::RecBranches*)
   {
      ::RecBranches *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::RecBranches >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("RecBranches", ::RecBranches::Class_Version(), "ROOTBranches.h", 44,
                  typeid(::RecBranches), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::RecBranches::Dictionary, isa_proxy, 4,
                  sizeof(::RecBranches) );
      instance.SetNew(&new_RecBranches);
      instance.SetNewArray(&newArray_RecBranches);
      instance.SetDelete(&delete_RecBranches);
      instance.SetDeleteArray(&deleteArray_RecBranches);
      instance.SetDestructor(&destruct_RecBranches);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::RecBranches*)
   {
      return GenerateInitInstanceLocal(static_cast<::RecBranches*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::RecBranches*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_GenBranches(void *p = nullptr);
   static void *newArray_GenBranches(Long_t size, void *p);
   static void delete_GenBranches(void *p);
   static void deleteArray_GenBranches(void *p);
   static void destruct_GenBranches(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::GenBranches*)
   {
      ::GenBranches *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::GenBranches >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("GenBranches", ::GenBranches::Class_Version(), "ROOTBranches.h", 109,
                  typeid(::GenBranches), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::GenBranches::Dictionary, isa_proxy, 4,
                  sizeof(::GenBranches) );
      instance.SetNew(&new_GenBranches);
      instance.SetNewArray(&newArray_GenBranches);
      instance.SetDelete(&delete_GenBranches);
      instance.SetDeleteArray(&deleteArray_GenBranches);
      instance.SetDestructor(&destruct_GenBranches);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::GenBranches*)
   {
      return GenerateInitInstanceLocal(static_cast<::GenBranches*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::GenBranches*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr EventBranches::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *EventBranches::Class_Name()
{
   return "EventBranches";
}

//______________________________________________________________________________
const char *EventBranches::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::EventBranches*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int EventBranches::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::EventBranches*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *EventBranches::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::EventBranches*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *EventBranches::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::EventBranches*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr RecBranches::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *RecBranches::Class_Name()
{
   return "RecBranches";
}

//______________________________________________________________________________
const char *RecBranches::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RecBranches*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int RecBranches::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RecBranches*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RecBranches::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RecBranches*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RecBranches::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RecBranches*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr GenBranches::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *GenBranches::Class_Name()
{
   return "GenBranches";
}

//______________________________________________________________________________
const char *GenBranches::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::GenBranches*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int GenBranches::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::GenBranches*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *GenBranches::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::GenBranches*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *GenBranches::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::GenBranches*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
void EventBranches::Streamer(TBuffer &R__b)
{
   // Stream an object of class EventBranches.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(EventBranches::Class(),this);
   } else {
      R__b.WriteClassBuffer(EventBranches::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_EventBranches(void *p) {
      return  p ? new(p) ::EventBranches : new ::EventBranches;
   }
   static void *newArray_EventBranches(Long_t nElements, void *p) {
      return p ? new(p) ::EventBranches[nElements] : new ::EventBranches[nElements];
   }
   // Wrapper around operator delete
   static void delete_EventBranches(void *p) {
      delete (static_cast<::EventBranches*>(p));
   }
   static void deleteArray_EventBranches(void *p) {
      delete [] (static_cast<::EventBranches*>(p));
   }
   static void destruct_EventBranches(void *p) {
      typedef ::EventBranches current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::EventBranches

//______________________________________________________________________________
void RecBranches::Streamer(TBuffer &R__b)
{
   // Stream an object of class RecBranches.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RecBranches::Class(),this);
   } else {
      R__b.WriteClassBuffer(RecBranches::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_RecBranches(void *p) {
      return  p ? new(p) ::RecBranches : new ::RecBranches;
   }
   static void *newArray_RecBranches(Long_t nElements, void *p) {
      return p ? new(p) ::RecBranches[nElements] : new ::RecBranches[nElements];
   }
   // Wrapper around operator delete
   static void delete_RecBranches(void *p) {
      delete (static_cast<::RecBranches*>(p));
   }
   static void deleteArray_RecBranches(void *p) {
      delete [] (static_cast<::RecBranches*>(p));
   }
   static void destruct_RecBranches(void *p) {
      typedef ::RecBranches current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::RecBranches

//______________________________________________________________________________
void GenBranches::Streamer(TBuffer &R__b)
{
   // Stream an object of class GenBranches.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(GenBranches::Class(),this);
   } else {
      R__b.WriteClassBuffer(GenBranches::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_GenBranches(void *p) {
      return  p ? new(p) ::GenBranches : new ::GenBranches;
   }
   static void *newArray_GenBranches(Long_t nElements, void *p) {
      return p ? new(p) ::GenBranches[nElements] : new ::GenBranches[nElements];
   }
   // Wrapper around operator delete
   static void delete_GenBranches(void *p) {
      delete (static_cast<::GenBranches*>(p));
   }
   static void deleteArray_GenBranches(void *p) {
      delete [] (static_cast<::GenBranches*>(p));
   }
   static void destruct_GenBranches(void *p) {
      typedef ::GenBranches current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::GenBranches

namespace ROOT {
   // Registration Schema evolution read functions
   int RecordReadRules_libROOTBranchesDict() {
      return 0;
   }
   static int _R__UNIQUE_DICT_(ReadRules_libROOTBranchesDict) = RecordReadRules_libROOTBranchesDict();R__UseDummy(_R__UNIQUE_DICT_(ReadRules_libROOTBranchesDict));
} // namespace ROOT
namespace {
  void TriggerDictionaryInitialization_libROOTBranchesDict_Impl() {
    static const char* headers[] = {
"/work/clas12/storyf/SF_analysis_software_v2.0/include/ROOTBranches.h",
nullptr
    };
    static const char* includePaths[] = {
"/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/root/6.36.04/include",
"/work/clas12/storyf/SF_analysis_software_v2.0/include",
"/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/clas12root/1.8.6b/4.3.0/hipo4",
"/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/clas12root/1.8.6b/4.3.0/Clas12Banks",
"/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/clas12root/1.8.6b/4.3.0/Clas12Root",
"/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/hipo/4.3.0/include",
"/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/ccdb/1.99.7/include",
"/u/scigroup/cvmfs/hallb/clas12/sw/noarch/rcdb/1.99.7/cpp/include",
"/u/scigroup/cvmfs/hallb/clas12/sw/noarch/clas12-qadb/3.4.0/srcC/include",
"/work/clas12/storyf/SF_analysis_software_v2.0",
"/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/root/6.36.04/include/",
"/w/hallb-scshelf2102/clas12/storyf/SF_analysis_software_v2.0/build/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "libROOTBranchesDict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
struct __attribute__((annotate("$clingAutoload$/work/clas12/storyf/SF_analysis_software_v2.0/include/ROOTBranches.h")))  EventBranches;
struct __attribute__((annotate("$clingAutoload$/work/clas12/storyf/SF_analysis_software_v2.0/include/ROOTBranches.h")))  RecBranches;
struct __attribute__((annotate("$clingAutoload$/work/clas12/storyf/SF_analysis_software_v2.0/include/ROOTBranches.h")))  GenBranches;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "libROOTBranchesDict dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "/work/clas12/storyf/SF_analysis_software_v2.0/include/ROOTBranches.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"EventBranches", payloadCode, "@",
"GenBranches", payloadCode, "@",
"RecBranches", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("libROOTBranchesDict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_libROOTBranchesDict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_libROOTBranchesDict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_libROOTBranchesDict() {
  TriggerDictionaryInitialization_libROOTBranchesDict_Impl();
}
