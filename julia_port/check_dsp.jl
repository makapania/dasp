using DSP

# Check what functions are available in DSP module
println("DSP module methods containing 'savitz':")
for name in names(DSP, all=true)
    if occursin("savitz", lowercase(string(name)))
        println("  ", name)
    end
end

println("\nDSP module methods containing 'golay':")
for name in names(DSP, all=true)
    if occursin("golay", lowercase(string(name)))
        println("  ", name)
    end
end

println("\nAll exported DSP functions:")
for name in names(DSP)
    println("  ", name)
end
