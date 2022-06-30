module FJSPT2
  export GameEnv, GameSpec
  include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec
    include("params.jl")
  end
end
