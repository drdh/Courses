## Julia 类型（jltype.c）

[TOC]

## Julia类型概览

### 基本类型

常见的就是`Float64`、`Float32`、`Int64`、`UInt64`等数据类型，可以自己使用关键字`primitive`声明

```julia
primitive type «name» «bits» end
primitive type «name» <: «supertype» «bits» end
```



### 抽象类型

是类型图中的节点，不能实例化，用于描述具体类型的集合，常见的有`Number`、`Real`、`Signed`等，可以使用关键字`abstract`声明

```julia
abstract type «name» end
abstract type «name» <: «supertype» end
```



### 复合类型

用户自定义类型，使用关键字`struct`声明，同时可以使用`::`对类型进行注释和约束

```julia
julia> struct Foo
           bar
           baz::Int
           qux::Float64
       end
```

特点：不可变，在构造后不可以修改

```julia
julia> struct Foo
       bar
       baz::Int
       qux::Float64
       end

julia> foo = Foo(1,2,3)
Foo(1, 2, 3.0)

julia> foo.bar
1

julia> foo.bar = 2
ERROR: type Foo is immutable
Stacktrace:
 [1] setproperty!(::Foo, ::Symbol, ::Int64) at ./sysimg.jl:19
 [2] top-level scope at none:0

```



### 可变复合类型

使用关键字`mutable`声明，构造后可变

```julia
julia> mutable struct Foo1
       bar
       baz::Int
       qux::Float64
       end

julia> foo = Foo1(1,2,3)
Foo1(1, 2, 3.0)

julia> foo.bar = 2
2

```



### Union类型

类型的联合

```julia
julia> IntOrString = Union{Int,AbstractString}
Union{Int64, AbstractString}

julia> 1 :: IntOrString
1

julia> "Hello!" :: IntOrString
"Hello!"
```



### 参数类型

在类型中加入参数，为泛型编程引入。有参数化符合类型，参数化抽象类型。



### 参数化复合类型

在类型名称后的`{}`中引入参数

```julia
julia> struct Point{T}
           x::T
           y::T
       end
```

而在成为具体类型时，只要将T用类型代替即可

```julia
julia> Point{Float64}
Point{Float64}

julia> Point{Real}
Point{Real}
// 都是Point的子类
// 但是虽然有Float64<:Real,但是Point{Float64}不是Point{Real}的子类，也就是说类型参数是不变的，不是协变也不是逆变
```



### 参数化抽象类型

与参数化复合类型类似

```julia
julia> abstract type Pointy{T} end
```

则此时`Point`类型是所有类型代替T后的具体类型的父类了



### 元组类型

是函数参数的抽象，使用`Tuple`声明，如`Tuple{Int,Real}`



### Vararg元组类型

最后一个参数为特殊类型`Vararg`的元组，表示一定或者任意数量的尾随元素

```julia
julia> Tuple{Vararg{Int,3}}
Tuple{Int64,Int64,Int64}

julia> Tuple{AbstractString,Vararg{Int}}
Tuple{AbstractString,Vararg{Int64,N} where N}
```



### NamedTuple类型

有两个参数：一个给出字段名称的符号元组，一个给出字段类型的元组类型。 

```julia
julia> typeof((a=1,b="hello"))
NamedTuple{(:a, :b),Tuple{Int64,String}}
```



### UnionAll类型

是为了参数化而设，在实例化前，并不能知道形如`Array{T} where T<:Integer `的具体类型，只知道是T为`Interger`的子类的所有实例化`Array`的集合所以使用`UnionAll`类型来对这样的类型进行描述，可以有上限和下限的约束。

### Singleton类型

`Type{T}`的唯一实例是`T`

```julia
julia> isa(Float64, Type{Float64})
true

julia> isa(Real, Type{Float64})
false

julia> isa(Real, Type{Real})
true

julia> isa(Float64, Type{Real})
false
```

而`Type`是一个抽象类型，将所有的类型对象作为其实例

```julia
julia> isa(Type{Float64}, Type)
true

julia> isa(Float64, Type)
true

julia> isa(Type, Type)
true
```



## Julia  内置类型

### 一些内置的基本类型

从`jltype.c`文件`line:53-line:68`的定义来看，有以下内置的基本类型（但其实应该不止这些，后面用julia的subtypes函数可以看到更多的内置基本类型）

```c
jl_datatype_t *jl_bool_type;
jl_datatype_t *jl_char_type;
jl_datatype_t *jl_int8_type;
jl_datatype_t *jl_uint8_type;
jl_datatype_t *jl_int16_type;
jl_datatype_t *jl_uint16_type;
jl_datatype_t *jl_int32_type;
jl_datatype_t *jl_uint32_type;
jl_datatype_t *jl_int64_type;
jl_datatype_t *jl_uint64_type;
jl_datatype_t *jl_float16_type;
jl_datatype_t *jl_float32_type;
jl_datatype_t *jl_float64_type;
```

官方文档给出的一些内置原始类型：

```julia
primitive type Float16 <: AbstractFloat 16 end
primitive type Float32 <: AbstractFloat 32 end
primitive type Float64 <: AbstractFloat 64 end

primitive type Bool <: Integer 8 end
primitive type Char <: AbstractChar 32 end

primitive type Int8    <: Signed   8 end
primitive type UInt8   <: Unsigned 8 end
primitive type Int16   <: Signed   16 end
primitive type UInt16  <: Unsigned 16 end
primitive type Int32   <: Signed   32 end
primitive type UInt32  <: Unsigned 32 end
primitive type Int64   <: Signed   64 end
primitive type UInt64  <: Unsigned 64 end
primitive type Int128  <: Signed   128 end
primitive type UInt128 <: Unsigned 128 end
```



### 一些内置的抽象类型

考虑到数据的层次关系，也内置了一些抽象的数据类型(抽象类型不能实例化）

```julia
abstract type Number end
abstract type Real     <: Number end
abstract type AbstractFloat <: Real end
abstract type Integer  <: Real end
abstract type Signed   <: Integer end
abstract type Unsigned <: Integer end
```



### 使用Julia的`subtypes()`函数看内置的类型

```julia
julia> function check_all_subtypes(T, space = 0)
	       println("\t" ^ space, T)
    	   for t in subtypes(T)
       			if t != Any
       				check_all_subtypes(t, space+1)
       			end
       		end
       end
check_all_subtypes (generic function with 2 methods)

julia> check_all_subtypes(Real)
Real
	AbstractFloat
		BigFloat
		Float16
		Float32
		Float64
	AbstractIrrational
		Irrational
	Integer
		Bool
		Signed
			BigInt
			Int128
			Int16
			Int32
			Int64
			Int8
		Unsigned
			UInt128
			UInt16
			UInt32
			UInt64
			UInt8
	Rational
```



## Union

### 特点概括

`Union`类型有以下特点：

- 当`Union`内只有一种类型时，`Union`无效
- 在`Union`中会进行子类型判断，去除子类
- 当有嵌套存在时，会对嵌套的`Union`展开

```shell
julia> Union{Int32,Int64} //声明Union
Union{Int32,Int64}

julia> Union{Int64}	//当Union内只有一种类型时，Union无效
Int64

julia> Union{Int32,Int64,Real} //在Union中会进行子类型判断，去除子类
Real

julia> Union{Int32,Int64,Union{Int64}}	//当有嵌套存在时，会对嵌套的Union展开
Union{Int32,Int64}

julia> Union{Int32,Int64,Union{Int64,Int32}} 
Union{Int32,Int64}

julia> Union{Int,Type{Int64},Type{Int32}}
Union{Int64,Type{Int32},Type{Int64}}
```



### 具体实现

```c
typedef struct _jl_datatype_t {
    JL_DATA_TYPE
    jl_typename_t *name;
    struct _jl_datatype_t *super;
    jl_svec_t *parameters;
    jl_svec_t *types;
    jl_svec_t *names;
    jl_value_t *instance;  // for singletons
    const jl_datatype_layout_t *layout;
    int32_t size; // TODO: move to _jl_datatype_layout_t
    int32_t ninitialized;
    uint32_t uid;
    uint8_t abstract;
    uint8_t mutabl;
    // memoized properties
    uint8_t hasfreetypevars; // majority part of isconcrete computation
    uint8_t isconcretetype; // whether this type can have instances
    uint8_t isdispatchtuple; // aka isleaftupletype
    uint8_t isbitstype; // relevant query for C-api and type-parameters
    uint8_t zeroinit; // if one or more fields requires zero-initialization
    uint8_t isinlinealloc; // if this is allocated inline
    void *struct_decl;  //llvm::Type*
    void *ditype; // llvm::MDNode* to be used as llvm::DIType(ditype)
} jl_datatype_t;

typedef struct {
    JL_DATA_TYPE
    jl_value_t *a;
    jl_value_t *b;
} jl_uniontype_t; //Union是由所有已有类型链接起来的结果

//计数union的组成个数，会对嵌套的Union进行展开计数
static int count_union_components(jl_value_t **types, size_t n);

//返回第*pi层的类型
static jl_value_t *nth_union_component(jl_value_t *v, int *pi) JL_NOTSAFEPOINT

// 找到needle在hastack的第几层
int jl_find_union_component(jl_value_t *haystack, jl_value_t *needle, unsigned *nth);

//把Union展开，例如Union{Int,Union{Int,Float}}展开为out[]={Int,Int,Float}
static void flatten_type_union(jl_value_t **types, size_t n, jl_value_t **out, size_t *idx) 

//类型比较需要的一些函数

//module名称的比较
STATIC_INLINE const char *datatype_module_name(jl_value_t *t) JL_NOTSAFEPOINT

//数字的比较
STATIC_INLINE int cmp_(int a, int b) JL_NOTSAFEPOINT

// 类型的比较
int datatype_name_cmp(jl_value_t *a, jl_value_t *b) JL_NOTSAFEPOINT
{
    if (!jl_is_datatype(a))
        return jl_is_datatype(b) ? 1 : 0;
    if (!jl_is_datatype(b))
        return -1;
    int cmp = strcmp(str_(datatype_module_name(a)), str_(datatype_module_name(b))); //比较module名称
    if (cmp != 0)
        return cmp;
    cmp = strcmp(str_(jl_typename_str(a)), str_(jl_typename_str(b)));//比较类型的名称
    if (cmp != 0)
        return cmp;
    cmp = cmp_(jl_nparams(a), jl_nparams(b)); //名称相同则比较类型参数个数
    if (cmp != 0)
        return cmp;
    // compare up to 3 type parameters。只比较前三个类型参数，应该是出于效率考虑
    for (int i = 0; i < 3 && i < jl_nparams(a); i++) {
        jl_value_t *ap = jl_tparam(a, i);
        jl_value_t *bp = jl_tparam(b, i);
        if (ap == bp) {
            continue;
        }
        else if (jl_is_datatype(ap) && jl_is_datatype(bp)) {
            cmp = datatype_name_cmp(ap, bp);
            if (cmp != 0)
                return cmp;
        }
        else if (jl_is_unionall(ap) && jl_is_unionall(bp)) {
            cmp = datatype_name_cmp(jl_unwrap_unionall(ap), jl_unwrap_unionall(bp));
            if (cmp != 0)
                return cmp;
        }
        else {
            // give up
            cmp = 0;
        }
    }
    return cmp;
}


// sort singletons first, then DataTypes, then UnionAlls,
// ties broken alphabetically including module name & type parameters
// 输入给qsort的cmp函数，用来对Union里的类型进行排序
// 比如在Union{Int64,Int32}经过后会成为Union{Int32,Int64}
int union_sort_cmp(const void *ap, const void *bp) JL_NOTSAFEPOINT
{
    jl_value_t *a = *(jl_value_t**)ap;
    jl_value_t *b = *(jl_value_t**)bp;
    if (a == NULL)
        return b == NULL ? 0 : 1;
    if (b == NULL)
        return -1;
    if (jl_is_datatype(a)) {
        if (!jl_is_datatype(b))
            return -1;
        if (jl_is_datatype_singleton((jl_datatype_t*)a)) {
            if (jl_is_datatype_singleton((jl_datatype_t*)b))
                return datatype_name_cmp(a, b);
            return -1;
        }
        else if (jl_is_datatype_singleton((jl_datatype_t*)b)) {
            return 1;
        }
        else if (jl_isbits(a)) {
            if (jl_isbits(b))
                return datatype_name_cmp(a, b);
            return -1;
        }
        else if (jl_isbits(b)) {
            return 1;
        }
        else {
            return datatype_name_cmp(a, b);
        }
    }
    else {
        if (jl_is_datatype(b))
            return 1;
        return datatype_name_cmp(jl_unwrap_unionall(a), jl_unwrap_unionall(b));
    }
}

// 在新增一个Union时调用
// 例如Union{Int64,Int32}=>Union{Int32,Int64}
// Union{Int64,Int32,Union{Int32,Int64}}=>Union{Int32,Int64}
JL_DLLEXPORT jl_value_t *jl_type_union(jl_value_t **ts, size_t n)
{
    if (n == 0) return (jl_value_t*)jl_bottom_type;
    size_t i;
    for(i=0; i < n; i++) {	//校验参数，否则抛出error
        jl_value_t *pi = ts[i];
        if (!(jl_is_type(pi) || jl_is_typevar(pi)) || jl_is_vararg_type(pi))
            jl_type_error_rt("Union", "parameter", (jl_value_t*)jl_type_type, pi);
    }
    if (n == 1) return ts[0];	//当参数为一个时，Union无效。Union{Int64}即Int64
    size_t nt = count_union_components(ts, n);
    jl_value_t **temp;
    JL_GC_PUSHARGS(temp, nt+1);
    size_t count = 0;
    flatten_type_union(ts, n, temp, &count);
    assert(count == nt);
    size_t j;
    // 在for循环中对flatten后的结果temp进行去重和子类判断
    for(i=0; i < nt; i++) {
        int has_free = temp[i]!=NULL && jl_has_free_typevars(temp[i]);
        for(j=0; j < nt; j++) {
            if (j != i && temp[i] && temp[j]) {
                if (temp[i] == temp[j] || temp[i] == jl_bottom_type ||
                    temp[j] == (jl_value_t*)jl_any_type ||
                    (!has_free && !jl_has_free_typevars(temp[j]) &&
                     jl_subtype(temp[i], temp[j]))) { 
                    //判断子类是为了达到Union{Int64,Real}=>Real的目的
                    temp[i] = NULL;
                }
            }
        }
    }
    qsort(temp, nt, sizeof(jl_value_t*), union_sort_cmp);
    jl_value_t **ptu = &temp[nt];
    *ptu = jl_bottom_type;
    int k;
    for (k = (int)nt-1; k >= 0; --k) {
        if (temp[k] != NULL) {
            if (*ptu == jl_bottom_type)
                *ptu = temp[k];
            else
                *ptu = jl_new_struct(jl_uniontype_type, temp[k], *ptu);//将flatten后、去重、子类判断结束后剩余的类型，重新组成一个独一的Union类
        }
    }
    assert(*ptu != NULL);
    jl_value_t *tu = *ptu;
    JL_GC_POP();
    return tu;
}

typedef struct {
    JL_DATA_TYPE
    jl_value_t *a;
    jl_value_t *b;
} jl_uniontype_t; //Union是由所有已有类型链接起来的结果
```



## Type Cache

### Cache规则

类型Cache建立规则

- 分为`cache`和`linecache`，分别用于缓存可排序的类型和不能排序的类型
- 排序比较策略：对于基本类型，比较类型参数个数、`uid`、`object_id`；非基本类型，比较`uid`，名称`hash`值，排序方法使用快排
- 查找策略：在`cache`中使用二分查找，在`linecache`中顺序查找
- 插入策略：先确定类型是放入`cache`还是`linecache`，然后进行查找，未找到则返回一个给新类型插入的位置下标，然后进行插入



### 具体实现

```c
typedef struct {
    JL_DATA_TYPE
    jl_sym_t *name;
    struct _jl_module_t *module;
    jl_svec_t *names;  // field names
    // `wrapper` is either the only instantiation of the type (if no parameters)
    // or a UnionAll accepting parameters to make an instantiation.
    jl_value_t *wrapper;
    jl_svec_t *cache;        // sorted array，对于cacheable的类型，排序后放入cache
    jl_svec_t *linearcache;  // unsorted array，存放uncacheable的类型
    intptr_t hash;
    struct _jl_methtable_t *mt;
} jl_typename_t;

// ordered comparison of types
static int typekey_compare(jl_datatype_t *tt, jl_value_t **key, size_t n) JL_NOTSAFEPOINT
{
    size_t j;
    if (tt == NULL) return -1;  // place NULLs at end to allow padding for fast growing
    size_t tnp = jl_nparams(tt);
    if (n < tnp) return -1;		//类型参数少的在前
    if (n > tnp) return 1;
    for(j=0; j < n; j++) {
        jl_value_t *kj = key[j], *tj = jl_svecref(tt->parameters,j);
        if (tj != kj) {
            int dtk = jl_is_datatype(kj);
            if (!jl_is_datatype(tj)) {
                if (dtk) return 1;
                uint32_t tid = wrapper_id(tj), kid = wrapper_id(kj);
                if (kid != tid)	//uid小的在前
                    return kid < tid ? -1 : 1;
                if (tid)
                    continue;
                if (jl_egal(tj, kj))
                    continue;
                return (jl_object_id(kj) < jl_object_id(tj) ? -1 : 1);
                			//object_id小的在前
            }
            else if (!dtk) {
                return -1;
            }
            //不是基本的数据类型，则比较uid，然后比较名字hash值，都一样则递归比较
            assert(dtk && jl_is_datatype(tj));
            jl_datatype_t *dt = (jl_datatype_t*)tj;
            jl_datatype_t *dk = (jl_datatype_t*)kj;
            if (dk->uid != dt->uid) {
                return dk->uid < dt->uid ? -1 : 1;
            }
            else if (dk->uid != 0) {
                assert(0);
            }
            else if (dk->name->hash != dt->name->hash) {
                return dk->name->hash < dt->name->hash ? -1 : 1;
            }
            else {
                int cmp = typekey_compare(dt, jl_svec_data(dk->parameters), jl_nparams(dk));
                if (cmp != 0)
                    return cmp;
            }
        }
    }
    return 0;
}

static int dt_compare(const void *ap, const void *bp) JL_NOTSAFEPOINT
{
    jl_datatype_t *a = *(jl_datatype_t**)ap;
    jl_datatype_t *b = *(jl_datatype_t**)bp;
    if (a == b) return 0;
    if (b == NULL) return -1;
    if (a == NULL) return 1;
    return typekey_compare(b, jl_svec_data(a->parameters), jl_svec_len(a->parameters));
}

void jl_resort_type_cache(jl_svec_t *c)
{
    qsort(jl_svec_data(c), jl_svec_len(c), sizeof(jl_value_t*), dt_compare);
}

// 查找策略：对于可排序的type，在cache中使用二分法查。否则在linecache中顺序查找
static ssize_t lookup_type_idx(jl_typename_t *tn, jl_value_t **key, size_t n, int ordered)
{
    if (n==0) return -1;
    if (ordered) {	//在已排序的cache中查找
        jl_svec_t *cache = tn->cache;
        jl_datatype_t **data = (jl_datatype_t**)jl_svec_data(cache);
        size_t cl = jl_svec_len(cache);
        ssize_t lo = -1;
        ssize_t hi = cl;
        while (lo < hi-1) {	//用二分法查找
            ssize_t m = ((size_t)(lo+hi))>>1;
            int cmp = typekey_compare(data[m], key, n);
            if (cmp > 0)
                lo = m;
            else
                hi = m;
        }
        /*
          When a module is replaced, the new versions of its types are different but
          cannot be distinguished by typekey_compare, since the TypeNames can only
          be distinguished by addresses, which don't have a reliable order. So we
          need to allow sequences of typekey_compare-equal types in the ordered cache.
        */
        while (hi < cl && typekey_compare(data[hi], key, n) == 0) {
            if (typekey_eq(data[hi], key, n))
                return hi;
            hi++;
        }
        return ~hi;
    }
    else {
        jl_svec_t *cache = tn->linearcache; //在未排序的linecache里进行查找
        jl_datatype_t **data = (jl_datatype_t**)jl_svec_data(cache);
        size_t cl = jl_svec_len(cache);
        ssize_t i;
        for(i=0; i < cl; i++) {
            jl_datatype_t *tt = data[i];
            if (tt == NULL) return ~i;
            if (typekey_eq(tt, key, n))
                return i;
        }
        return ~cl; //未找到则返回给新类型存放的位置
    }
}


static void cache_insert_type(jl_value_t *type, ssize_t insert_at, int ordered)
{
    assert(jl_is_datatype(type));
    // assign uid if it hasn't been done already
    if (!jl_is_abstracttype(type) && ((jl_datatype_t*)type)->uid==0)
        ((jl_datatype_t*)type)->uid = jl_assign_type_uid();
    jl_svec_t *cache;
    if (ordered)
        // 可排序用cache
        cache = ((jl_datatype_t*)type)->name->cache;
    else
        // 不可排序用linecache
        cache = ((jl_datatype_t*)type)->name->linearcache;
    assert(jl_is_svec(cache));
    size_t n = jl_svec_len(cache);
    if (n==0 || jl_svecref(cache,n-1) != NULL) {
        jl_svec_t *nc = jl_alloc_svec(n < 8 ? 8 : (n*3)>>1);
        memcpy(jl_svec_data(nc), jl_svec_data(cache), sizeof(void*) * n);
        if (ordered)
            ((jl_datatype_t*)type)->name->cache = nc;
        else
            ((jl_datatype_t*)type)->name->linearcache = nc;
        jl_gc_wb(((jl_datatype_t*)type)->name, nc);
        cache = nc;
        n = jl_svec_len(nc);
    }
    jl_value_t **p = jl_svec_data(cache);
    size_t i = insert_at;
    jl_value_t *temp = p[i], *temp2;
    jl_svecset(cache, insert_at, (jl_value_t*)type);
    assert(i < n-1 || temp == NULL);
    while (temp != NULL && i < n-1) {
        i++;
        temp2 = p[i];
        p[i] = temp;
        temp = temp2;
    }
}

//插入策略
jl_value_t *jl_cache_type_(jl_datatype_t *type)
{
    if (is_cacheable(type)) {
        JL_TIMING(TYPE_CACHE_INSERT);
        int ord = is_typekey_ordered(jl_svec_data(type->parameters), jl_svec_len(type->parameters));
        ssize_t idx = lookup_type_idx(type->name, jl_svec_data(type->parameters),
                                      jl_svec_len(type->parameters), ord);
        if (idx >= 0)
            type = (jl_datatype_t*)jl_svecref(ord ? type->name->cache : type->name->linearcache, idx);
        else
            cache_insert_type((jl_value_t*)type, ~idx, ord);
    }
    return (jl_value_t*)type;
}

```



## Type Instantiation 

### Instatiation 流程

- 使用一个`stack`用于保存正在`Instantiate`的类型记录
- 先在`Cache`中查找是否已经实例化并已在缓存中，再在`stack`中查找是否已在处理过程中
- 对于`Tuple`，如果是`Tuple{...}`，那么就是任意的元组类型`AnyTuple`，如果是`Tuple{}`，那么就是空元组，即`EmptyTuple`，否则将`Vararg`类型展开，将`Tuple`实例化
- 开始正式的实例化，新建`jl_datatype_t`并根据已知的信息将此结构体内的元素进行初始化

```c
typedef struct _jl_datatype_t {
    JL_DATA_TYPE
    jl_typename_t *name;
    struct _jl_datatype_t *super;
    jl_svec_t *parameters;
    jl_svec_t *types;
    jl_svec_t *names;
    jl_value_t *instance;  // for singletons
    const jl_datatype_layout_t *layout;
    int32_t size; // TODO: move to _jl_datatype_layout_t
    int32_t ninitialized;
    uint32_t uid;
    uint8_t abstract;
    uint8_t mutabl;
    // memoized properties
    uint8_t hasfreetypevars; // majority part of isconcrete computation
    uint8_t isconcretetype; // whether this type can have instances
    uint8_t isdispatchtuple; // aka isleaftupletype
    uint8_t isbitstype; // relevant query for C-api and type-parameters
    uint8_t zeroinit; // if one or more fields requires zero-initialization
    uint8_t isinlinealloc; // if this is allocated inline
    void *struct_decl;  //llvm::Type*
    void *ditype; // llvm::MDNode* to be used as llvm::DIType(ditype)
} jl_datatype_t;
```

### 代码片段

```c
    // create and initialize new type
    ndt = jl_new_uninitialized_datatype();
    // associate these parameters with the new type on
    // the stack, in case one of its field types references it.
    top.tt = (jl_datatype_t*)ndt;
    top.prev = stack;
    stack = &top; //在stack中加入当前处理的新类型
    ndt->name = tn;
    jl_gc_wb(ndt, ndt->name);
    ndt->super = NULL; // 设置父类型
    ndt->parameters = p; 
    jl_gc_wb(ndt, ndt->parameters);
    ndt->types = NULL; // to be filled in below
    if (istuple) { //是否为Tuple
        ndt->types = p;
    }
    else if (isnamedtuple) { //是否为NamedTuple
        jl_value_t *names_tup = jl_svecref(p, 0);
        jl_value_t *values_tt = jl_svecref(p, 1);
        if (!jl_has_free_typevars(names_tup) && !jl_has_free_typevars(values_tt)) {
            if (!jl_is_tuple(names_tup))
                jl_type_error_rt("NamedTuple", "names", (jl_value_t*)jl_anytuple_type, names_tup);
            size_t nf = jl_nfields(names_tup);
            jl_svec_t *names = jl_alloc_svec_uninit(nf);
            for (size_t i = 0; i < nf; i++) {
                jl_value_t *ni = jl_fieldref(names_tup, i);
                if (!jl_is_symbol(ni))
                    jl_type_error_rt("NamedTuple", "name", (jl_value_t*)jl_symbol_type, ni);
                for (size_t j = 0; j < i; j++) {
                    if (ni == jl_svecref(names, j))
                        jl_errorf("duplicate field name in NamedTuple: \"%s\" is not unique", jl_symbol_name((jl_sym_t*)ni));
                }
                jl_svecset(names, i, ni);
            }
            if (!jl_is_datatype(values_tt))
                jl_error("NamedTuple field type must be a tuple type");
            if (jl_is_va_tuple((jl_datatype_t*)values_tt) || jl_nparams(values_tt) != nf)
                jl_error("NamedTuple names and field types must have matching lengths");
            ndt->names = names;
            jl_gc_wb(ndt, ndt->names);
            ndt->types = ((jl_datatype_t*)values_tt)->parameters;
            jl_gc_wb(ndt, ndt->types);
        }
        else {
            ndt->types = jl_emptysvec;
        }
    }
    ndt->mutabl = dt->mutabl;
    ndt->abstract = dt->abstract;
    ndt->instance = NULL;
    ndt->uid = 0;
    ndt->struct_decl = NULL;
    ndt->ditype = NULL;
    ndt->size = 0;
    jl_precompute_memoized_dt(ndt);

    // assign uid as early as possible
    if (cacheable && !ndt->abstract)
        ndt->uid = jl_assign_type_uid(); //为类型赋予uid

    if (istuple || isnamedtuple) {
        ndt->super = jl_any_type; //对于Tuple类型，其父类型为Any
    }
    else if (dt->super) {
        ndt->super = (jl_datatype_t*)inst_type_w_((jl_value_t*)dt->super, env, stack, 1);
        jl_gc_wb(ndt, ndt->super);
    }
    jl_svec_t *ftypes = dt->types;
    if (ftypes == NULL || dt->super == NULL) {
        // in the process of creating this type definition:
        // need to instantiate the super and types fields later
        assert(inside_typedef && !istuple && !isnamedtuple);
        arraylist_push(&partial_inst, ndt);
    }
    else {
        assert(ftypes != jl_emptysvec || jl_field_names(ndt) == jl_emptysvec || isnamedtuple);
        assert(ftypes == jl_emptysvec || !ndt->abstract);
        if (!istuple && !isnamedtuple) {
            // recursively instantiate the types of the fields
            ndt->types = inst_all(ftypes, env, stack, 1);
            jl_gc_wb(ndt, ndt->types);
        }
    }
    if (jl_is_primitivetype(dt)) {
        ndt->size = dt->size;
        ndt->layout = dt->layout;
        ndt->isbitstype = ndt->isinlinealloc = (!ndt->hasfreetypevars);
    }
    else if (cacheable && ndt->types != NULL && !ndt->abstract) {
        jl_compute_field_offsets(ndt);
    }

    if (istuple)
        ndt->ninitialized = ntp - isvatuple;
    else if (isnamedtuple)
        ndt->ninitialized = jl_svec_len(ndt->types);
    else
        ndt->ninitialized = dt->ninitialized;

    if (cacheable) {
        jl_cache_type_(ndt);
        JL_UNLOCK(&typecache_lock); // Might GC
    }

    JL_GC_POP();
    return (jl_value_t*)ndt;
```



## 自由类型变量和约束类型变量

- 自由类型变量，例如：`Array{T<:Any}`,此时`T`并没有类型约束，是一个自由类型变量
- 约束类型变量 `Array{Signed<:V<:Real}`，此时`V`的类型被约束为抽象类型`Singned`的父类，`Real`的子类。是一个约束的类型变量

```julia
julia> Union{T,S} where T<:Real where S<:Int
Union{T, S} where T<:Real where S<:Int64

julia> Int<:Union{T,S} where T<:Real where S<:Int
true
```



```C
typedef struct {
    JL_DATA_TYPE
    jl_sym_t *name;
    jl_value_t *lb;   // lower bound
    jl_value_t *ub;   // upper bound
} jl_tvar_t;

typedef struct jl_typeenv_t {
    jl_tvar_t *var;
    jl_value_t *val;
    struct jl_typeenv_t *prev;
} jl_typeenv_t;

// UnionAll type (iterated union over all values of a variable in certain bounds)
// written `body where lb<:var<:ub`
typedef struct {
    JL_DATA_TYPE
    jl_tvar_t *var;
    jl_value_t *body;
} jl_unionall_t;

typedef struct {
    JL_DATA_TYPE
    jl_value_t *a;
    jl_value_t *b;
} jl_uniontype_t; //Union是由所有已有类型链接起来的结果

static int typeenv_has(jl_typeenv_t *env, jl_tvar_t *v)
{
    while (env != NULL) {
        if (env->var == v)
            return 1;
        env = env->prev;
    }
    return 0;
}

//  UnionAll类型举例：Array{T} where T<:Integer
//  找自由类型变量
//  自由类型变量，例如：Array{T},此时T并没有约束，是一个自由类型变量
static int has_free_typevars(jl_value_t *v, jl_typeenv_t *env)
{
    if (jl_typeis(v, jl_tvar_type)) {	//普通类型变量，直接在类型环境中查找即可
        return !typeenv_has(env, (jl_tvar_t*)v);
    }
    if (jl_is_uniontype(v))	//对于Union类型，对组成Union的每个类型在类型环境中找
        return has_free_typevars(((jl_uniontype_t*)v)->a, env) ||
            has_free_typevars(((jl_uniontype_t*)v)->b, env);
    if (jl_is_unionall(v)) { 
        jl_unionall_t *ua = (jl_unionall_t*)v;
        jl_typeenv_t newenv = { ua->var, NULL, env };
        return has_free_typevars(ua->var->lb, env) || has_free_typevars(ua->var->ub, env) ||
            has_free_typevars(ua->body, &newenv);
    }
    if (jl_is_datatype(v)) { 
        int expect = ((jl_datatype_t*)v)->hasfreetypevars;
        if (expect == 0 || env == NULL)
            return expect;
        size_t i;
        for (i=0; i < jl_nparams(v); i++) {
            if (has_free_typevars(jl_tparam(v,i), env)) {
                assert(expect);
                return 1;
            }
        }
    }
    return 0;
}

// 找约束类型变量
//	Array{Signed<:V<:Real}，此时V就是一个约束的类型变量
static int jl_has_bound_typevars(jl_value_t *v, jl_typeenv_t *env)
{
    if (jl_typeis(v, jl_tvar_type))
        return typeenv_has(env, (jl_tvar_t*)v);
    if (jl_is_uniontype(v))
        return jl_has_bound_typevars(((jl_uniontype_t*)v)->a, env) ||
            jl_has_bound_typevars(((jl_uniontype_t*)v)->b, env);
    if (jl_is_unionall(v)) {
        jl_unionall_t *ua = (jl_unionall_t*)v;
        if (jl_has_bound_typevars(ua->var->lb, env) || jl_has_bound_typevars(ua->var->ub, env))
            return 1;
        jl_typeenv_t *te = env;
        while (te != NULL) {
            if (te->var == ua->var)
                break;
            te = te->prev;
        }
        if (te) te->var = NULL;  // temporarily remove this var from env
        int ans = jl_has_bound_typevars(ua->body, env);
        if (te) te->var = ua->var;
        return ans;
    }
    if (jl_is_datatype(v)) {
        if (!((jl_datatype_t*)v)->hasfreetypevars)
            return 0;
        size_t i;
        for (i=0; i < jl_nparams(v); i++) {
            if (jl_has_bound_typevars(jl_tparam(v,i), env))
                return 1;
        }
    }
    return 0;
}

```


