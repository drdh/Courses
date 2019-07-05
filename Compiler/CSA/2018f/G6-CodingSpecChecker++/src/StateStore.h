#pragma once

#include <map>
#include <set>
#include <vector>
#include <algorithm>

template <typename T>
struct TUCheckStateTrait
{
  typedef typename T::data_type data_type;
  // static inline void *MakeVoidPtr(data_type D) { return (void*) D; }
  static inline data_type *MakeData()
  {
    return new data_type();
  }

  static inline data_type *CastData(void *Ptr)
  {
    return static_cast<data_type *>(Ptr);
  }
};

#define REGISTER_TRAIT_WITH_TUCHECKSTATE(Name, Type) \
  namespace                                          \
  {                                                  \
  class Name                                         \
  {                                                  \
  };                                                 \
  using Name##Ty = Type;                             \
  }                                                  \
  template <>                                        \
  struct TUCheckStateTrait<Name>                     \
      : public TUCheckStatePartialTrait<Name##Ty>    \
  {                                                  \
    static void *GDMIndex()                          \
    {                                                \
      static int Index;                              \
      return &Index;                                 \
    }                                                \
  };

template <typename T>
struct TUCheckStatePartialTrait
{
};

// Partial-specialization for ImmutableMap.
template <typename Key, typename Data>
struct TUCheckStatePartialTrait<std::map<Key, Data>>
{
  using data_type = std::map<Key, Data>;
  using key_type = Key;
  using value_type = Data;
  using lookup_type = const value_type *;

  static data_type *MakeData()
  {
    return new data_type();
  }

  static inline data_type *CastData(void *Ptr)
  {
    return static_cast<data_type *>(Ptr);
  }

  static lookup_type Lookup(void *B, key_type K)
  {
    data_type *Map = static_cast<data_type *>(B);
    return &(Map->at(K));
  }

  static void Set(void *B, key_type K, value_type E)
  {
    data_type *Map = static_cast<data_type *>(B);
    (*Map)[K] = E;
  }

  static void Remove(void *B, key_type K)
  {
    data_type *Map = static_cast<data_type *>(B);
    Map->erase(K);
  }

  static bool Contains(void *B, key_type K)
  {
    data_type *Map = static_cast<data_type *>(B);
    return Map->find(K) != Map->end();
  }
};

// Partial-specialization for ImmutableMap.
template <typename Key>
struct TUCheckStatePartialTrait<std::set<Key>>
{
  using data_type = std::set<Key>;
  using key_type = Key;

  static data_type *MakeData()
  {
    return new data_type();
  }

  static inline data_type *CastData(void *Ptr)
  {
    return static_cast<data_type *>(Ptr);
  }

  static void Add(void *B, key_type K)
  {
    data_type *Data = static_cast<data_type *>(B);
    Data->insert(K);
  }

  static void Remove(void *B, key_type K)
  {
    data_type *Data = static_cast<data_type *>(B);
    Data->erase(K);
  }

  static bool Contains(void *B, key_type K)
  {
    data_type *Data = static_cast<data_type *>(B);
    return Data->find(K) != Data->end();
  }
};

// Partial-specialization for ImmutableMap.
template <typename T>
struct TUCheckStatePartialTrait<std::vector<T>>
{
  using data_type = std::vector<T>;
  using key_type = T;

  static data_type *MakeData()
  {
    return new data_type();
  }

  static inline data_type *CastData(void *Ptr)
  {
    return static_cast<data_type *>(Ptr);
  }

  static void Add(void *B, key_type K)
  {
    data_type *Data = static_cast<data_type *>(B);
    Data->push_back(K);
  }

  static void Remove(void *B, key_type K)
  {
    data_type *Data = static_cast<data_type *>(B);
    auto I = std::find(Data->begin(), Data->end(), K);
    Data->erase(I);
  }

  static bool Contains(void *B, key_type K)
  {
    data_type *Data = static_cast<data_type *>(B);
    return std::find(Data->begin(), Data->end(), K) != Data->end();
  }
};

#define TUCHECKSTATE_MAP(Key, Value) std::map<Key, Value>

#define REGISTER_MAP_WITH_TUCHECKSTATE(Name, Key, Value) \
  REGISTER_TRAIT_WITH_TUCHECKSTATE(Name, TUCHECKSTATE_MAP(Key, Value))

#define REGISTER_SET_WITH_TUCHECKSTATE(Name, Elem) \
  REGISTER_TRAIT_WITH_TUCHECKSTATE(Name, std::set<Elem>)

#define REGISTER_LIST_WITH_TUCHECKSTATE(Name, Elem) \
  REGISTER_TRAIT_WITH_TUCHECKSTATE(Name, std::vector<Elem>)

class TUCheckStateStore
{
private:
  std::map<void *, void *> GDM;

private:
  TUCheckStateStore() {}
  ~TUCheckStateStore();

public:
  static TUCheckStateStore &Get();

  template <typename T>
  void *FindGDM(void *K)
  {
    if (GDM.find(K) == GDM.end())
    {
      GDM[K] = TUCheckStateTrait<T>::MakeData();
    }
    return GDM[K];
  }

  template <typename T>
  void
  add(typename TUCheckStateTrait<T>::key_type Key)
  {
    void *Data = FindGDM<T>(TUCheckStateTrait<T>::GDMIndex());
    TUCheckStateTrait<T>::Add(Data, Key);
  }

  template <typename T>
  typename TUCheckStateTrait<T>::lookup_type
  get(typename TUCheckStateTrait<T>::key_type Key)
  {
    void *Data = FindGDM<T>(TUCheckStateTrait<T>::GDMIndex());
    return TUCheckStateTrait<T>::Lookup(Data, Key);
  }

  template <typename T>
  typename const TUCheckStateTrait<T>::data_type *
  get()
  {
    void *Data = FindGDM<T>(TUCheckStateTrait<T>::GDMIndex());
    return TUCheckStateTrait<T>::CastData(Data);
  }

  template <typename T>
  void
  remove(typename TUCheckStateTrait<T>::key_type Key)
  {
    void *Data = FindGDM<T>(TUCheckStateTrait<T>::GDMIndex());
    TUCheckStateTrait<T>::Remove(Data, Key);
  }

  template <typename T>
  void
  set(typename TUCheckStateTrait<T>::key_type Key,
      typename TUCheckStateTrait<T>::value_type Elem)
  {
    void *Data = FindGDM<T>(TUCheckStateTrait<T>::GDMIndex());
    TUCheckStateTrait<T>::Set(Data, Key, Elem);
  }

  template <typename T>
  bool contains(typename TUCheckStateTrait<T>::key_type key)
  {
    void *Data = FindGDM<T>(TUCheckStateTrait<T>::GDMIndex());
    return TUCheckStateTrait<T>::Contains(Data, key);
  }
};