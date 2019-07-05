
# Pretty printing
# ---------------
# Implement pshow(io::PrettyIO, t::T) for types T to be prettyprinted.

# pshow() should print no initial or final newlines.
# It should print using pshow and pprint.

# pprint(io, a, {b, c})
# prints a, then prints b and c indented.
# (The indenting is only noticable if there is a line break within {b, c})

# pprint(io, a, io->fun(io), b)
# invokes fun(io) on the current io. I e
# pprint(io, {a, io->fun(io), b})
# invokes fun() with an indented io.

#load("utils/utils.jl")

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
pprint(io::PrettyIO, args...) = (for arg in args; pprint(io, arg); end)

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
        #if nsp==0; nsp=8; end
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
function pshow_mainbody(io::PrettyIO, ex)
    if is_expr(ex, :block)
        args = ex.args
        for (arg, k) in enumerate(args)
            if !is_expr(arg, :line)
                pprint(io, "\n")
            end
            pshow(io, arg)
        end
    else
        if !is_expr(ex, :line);  pprint(io, "\n");  end
        pshow(io, ex)
    end
end

## show arguments of a block, and then body
pshow_body(io::PrettyIO, body) = pshow_body(io, {}, body)
function pshow_body(io::PrettyIO, arg, body)
    pprint(io, {arg, io->pshow_mainbody(io, body) })
end
function pshow_body(io::PrettyIO, args::Vector, body)
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
pshow_quoted_expr(io::PrettyIO, ex) =pprint(io, ":($ex)")


## show an expr
function pshow(io::PrettyIO, ex::Expr)
    const infix = {:(=)=>"=", :(.)=>".", doublecolon=>"::", :(:)=>":",
                   :(->)=>"->", :(=>)=>"=>",
                   :(&&)=>" && ", :(||)=>" || "}
    const parentypes = {:call=>("(",")"), :ref=>("[","]"), :curly=>("{","}")}

    head = ex.head
    args = ex.args
    nargs = length(args)

    if has(infix, head) && nargs==2             # infix operations
#        pprint(io, "(",{args[1], infix[head], args[2]},")")
        pprint(io, {args[1], infix[head], args[2]})
    elseif has(parentypes, head) && nargs >= 1  # :call/:ref/:curly
        pprint(io, args[1])
        pshow_comma_list(io, args[2:end], parentypes[head]...)
    elseif (head == :comparison) && (nargs>=3 && isodd(nargs)) # :comparison
        pprint("(",{args},")")
    elseif ((contains([:return, :abstract, :const] , head) && nargs==1) ||
            contains([:local, :global], head))
        pshow_comma_list(io, args, string(head)*" ", "")
    elseif head == :typealias && nargs==2
        pshow_delim_list(io, args, string(head)*" ", " ", "")
    elseif (head == :quote) && (nargs==1)       # :quote
        pshow_quoted_expr(io, args[1])
    elseif (head == :line) && (1 <= nargs <= 2) # :line
        let io=comment(io)
            if nargs == 1
                linecomment = "line "*string(args[1])*": "
            else
                @assert nargs==2
#               linecomment = "line "*string(args[1])*", "*string(args[2])*": "
                linecomment = string(args[2])*", line "*string(args[1])*": "
            end
            if str_fits_on_line(io, strlen(linecomment)+13)
                pprint(io, "\t#  ", linecomment)
            else
                pprint(io, "\n", linecomment)
            end
        end
    elseif head == :if && nargs == 3  # if/else
        pprint(io, 
            "if ", io->pshow_body(io, args[1], args[2]),
            "\nelse ", io->pshow_body(io, args[3]),
            "\nend")
    elseif head == :try && nargs == 3 # try[/catch]
        pprint(io, "try ", io->pshow_body(io, args[1]))
        if !(is(args[2], false) && is_expr(args[3], :block, 0))
            pprint(io, "\ncatch ", io->pshow_body(io, args[2], args[3]))
        end
        pprint(io, "\nend")
    elseif head == :let               # :let 
        pprint(io, "let ", 
            io->pshow_body(io, args[2:end], args[1]), "\nend")
    elseif head == :block
        pprint(io, "begin ", io->pshow_body(io, ex), "\nend")
    elseif contains([:for, :while, :function, :if, :type], head) && nargs == 2
        pprint(io, string(head), " ", 
            io->pshow_body(io, args[1], args[2]), "\nend")
    else
        pprint(io, head)
        pshow_comma_list(indent(io), args, "(", ")")
    end
end
