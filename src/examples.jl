module Examples

  using ..AlphaZero

  include("../games/tictactoe/main.jl")
  export Tictactoe

  include("../games/connect-four/main.jl")
  export ConnectFour

  include("../games/grid-world/main.jl")
  export GridWorld

  include("../games/mancala/main.jl")
  export Mancala

  include("../games/jsp/main.jl")
  export JSP

  include("../games/fjsp/main.jl")
  export FJSP

  include("../games/fjspt/main.jl")
  export FJSPT

  include("../games/fjspt2/main.jl")
  export FJSPT2

  include("../games/fjspt3/main.jl")
  export FJSPT3

  include("../games/dfjspt/main.jl")
  export DFJSPT


  const games = Dict(
    "grid-world" => GridWorld.Training.experiment.gspec,
    "tictactoe" => Tictactoe.Training.experiment.gspec,
    "connect-four" => ConnectFour.Training.experiment.gspec,
    "mancala" => Mancala.Training.experiment.gspec,
    "jsp" => JSP.Training.experiment.gspec,
    "fjsp" => FJSP.Training.experiment.gspec,
    "fjspt" => FJSPT.Training.experiment.gspec,
    "fjspt2" => FJSPT2.Training.experiment.gspec,
    "fjspt3" => FJSPT3.Training.experiment.gspec,
    "dfjspt" => DFJSPT.Training.experiment.gspec)
    # "ospiel_ttt" => OSpielTictactoe.GameSpec()
  # ospiel_ttt is added from openspiel_example.jl when OpenSpiel.jl is imported


  const experiments = Dict(
    "grid-world" => GridWorld.Training.experiment,
    "tictactoe" => Tictactoe.Training.experiment,
    "connect-four" => ConnectFour.Training.experiment,
    "mancala" => Mancala.Training.experiment,
    "jsp" => JSP.Training.experiment,
    "fjsp" => FJSP.Training.experiment,
    "fjspt" => FJSPT.Training.experiment,
    "fjspt2" => FJSPT2.Training.experiment,
    "fjspt3" => FJSPT3.Training.experiment,
    "dfjspt" => DFJSPT.Training.experiment)
    # "ospiel_ttt" => OSpielTictactoe.Training.experiment

end