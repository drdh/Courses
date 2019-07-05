
code = quote

is_expr(ex, head::Symbol) = (isa(ex, Expr) && (ex.head == head))
function is_expr(ex, head::Symbol, nargs::Int)
    is_expr(ex, head) && length(ex.args) == nargs
end


pprintln(args...) = pprint(args..., '\n')
pprint(args...) = pprint(default_pretty(), args...)
pshow(args...)  = pshow(default_pretty(), args...)

# fallback for io::IO
pprint(io::IO, args...) = print(io, args...) 


# -- PrettyIO -----------------------------------------------------------------

abstract PrettyIO

pprint_nowrap(io::PrettyIO, s::String) = (for c in s; pprint_nowrap(io, c);end)
function pprint(io::PrettyIO, s::String)
    n = strlen(s)
    if !str_fits_on_line(io, n)
        pprint(io, '\n')
    end
    for c in s; pprint(io, c); end
end
pprint(io::PrettyIO, arg::Any) = pshow(io, arg)
pprint(io::PrettyIO, args...) = foreach(arg->pprint(io, arg), args)

pshow(io::PrettyIO, arg::Any) = pprint(io, sshow(arg))


pprint(io::PrettyIO, v::Vector) = pprint(indent(io), v...)
pprint(io::PrettyIO, pprinter::Function) = pprinter(io)


comment(io::PrettyIO) = PrettyChild(io, ()->"# ")

indent(io::PrettyIO) = indent(io::PrettyIO, 4)
indent(io::PrettyIO, indent::Int) = (pre=" "^indent; PrettyChild(io, ()->pre))


# -- PrettyRoot ---------------------------------------------------------------

type PrettyRoot <: PrettyIO
    parent::IO
    width::Int

    currpos::Int
    autowrap::Bool

    function PrettyRoot(parent::IO, width::Int)
        if width < 1; error("width must be >= 1, got ", width); end
        new(parent, width, 0, false)
    end
end

function str_fits_on_line(io::PrettyRoot, n::Integer)
    (!io.autowrap) || (io.currpos+n <= io.width)
end

function pprint_nowrap(io::PrettyRoot, c::Char)
    if c=='\t'
        nsp::Int = (-io.currpos)&7
        if nsp==0; nsp=8; end
        print(io.parent, " "^nsp)
        io.currpos += nsp
    else
        print(io.parent, c)
        io.currpos += 1
    end
    if c == '\n'
        io.currpos = 0
        io.autowrap = false
        return true
    end
    return false
end
function pprint(io::PrettyRoot, c::Char)
    if 2*io.currpos < io.width; io.autowrap=true; end

    if pprint_nowrap(io, c); return true; end
    if io.autowrap && (io.currpos >= io.width)
        return pprint_nowrap(io, '\n')
    end
    return false
end

default_pretty() = PrettyRoot(OUTPUT_STREAM, 80)


# -- PrettyChild --------------------------------------------------------------

type PrettyChild <: PrettyIO
    parent::PrettyIO
    newline_hook::Function

    function PrettyChild(parent::PrettyIO, newline_hook::Function)
        new(parent, newline_hook)
    end
end

str_fits_on_line(io::PrettyChild, n::Integer) = str_fits_on_line(io.parent, n)

pprint_nowrap(io::PrettyChild, c::Char) = pprint_nowrap(io.parent, c)
function pprint(io::PrettyChild, c::Char)
    newline = pprint(io.parent, c)::Bool
    if newline
        pprint_nowrap(io.parent, io.newline_hook())
    end
    return newline
end


# == Expr prettyprinting ======================================================

const doublecolon = @eval (:(x::Int)).head

## list printing
function pshow_comma_list(io::PrettyIO, args::Vector, 
                          open::String, close::String) 
    pshow_delim_list(io, args, open, ", ", close)
end
function pshow_delim_list(io::PrettyIO, args::Vector, open::String, 
                          delim::String, close::String)
    pprint(io, {open, 
                io->pshow_list_delim(io, args, delim)},
           close)
end
function pshow_list_delim(io::PrettyIO, args::Vector, delim::String)
    for (arg, k) in enumerate(args)
        pshow(io, arg)
        if k < length(args)
            pprint(io, delim)
        end
    end
end

## show the body of a :block
pshow_mainbody(io::PrettyIO, ex) = pshow(io, ex)
function pshow_mainbody(io::PrettyIO, ex::Expr)
    if ex.head == :block
        args = ex.args
        for (arg, k) in enumerate(args)
            if !is_expr(arg, :line)
                pprint(io, "\n")
            end
            pshow(io, arg)
        end
    else
        pshow(io, ex)
    end
end

## show arguments of a block, and then body
pshow_body(io::PrettyIO, body::Expr) = pshow_body(io, {}, body)
function pshow_body(io::PrettyIO, arg, body::Expr)
    pprint(io, {arg, io->pshow_mainbody(io, body) })
end
function pshow_body(io::PrettyIO, args::Vector, body::Expr)
    pprint(io, {
            io->pshow_comma_list(io, args, "", ""), 
            io->pshow_mainbody(io, body)
        })
end

## show ex as if it were quoted
function pshow_quoted_expr(io::PrettyIO, sym::Symbol)
    if !is(sym,:(:)) && !is(sym,:(==))
        pprint(io, ":$sym")
    else
        pprint(io, ":($sym)")
    end
end
function pshow_quoted_expr(io::PrettyIO, ex::Expr)
    if ex.head == :block
        pprint(io, "quote ", io->pshow_body(io, ex), "\nend")
    else
        pprint(io, "quote(", {ex}, ")")
    end
end

end # quote

load("prettyshow.jl")

for ex in code.args
    pprintln(ex)
    if !is_expr(ex, :line)
        println()
    end
end

println()
fname = "unnecessarily_long_function_name"
pprintln("for ", {"i=1:n\n", 
             "for ", {"j=1:m\n",
                 "X[",{"i, j"},"] = ",
                     "A[",{"$fname(i)"},"]", " * ",
                     "B[",{"$fname(j)"},"]"
             }, "\nend"
         }, "\nend")

