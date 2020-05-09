$(document).ready(function () {
    $("#year_select_home").change(function(){
        var val = $(this).val();
        if (val == 19) {
            $("#hometeam").html(TeamsHtml19)
        }else if (val == 18) {
            $("#hometeam").html(TeamsHtml18)
        }else if (val == 17) {
            $("#hometeam").html(TeamsHtml17)
        }else if (val == 16) {
            $("#hometeam").html(TeamsHtml16)
        }else if (val == 15) {
            $("#hometeam").html(TeamsHtml15)
        }else if (val == 14) {
            $("#hometeam").html(TeamsHtml14)
        }else if (val == 13) {
            $("#hometeam").html(TeamsHtml13)
        }else if (val == 12) {
            $("#hometeam").html(TeamsHtml12)
          }
      })

    $("#year_select_away").change(function(){
        var val = $(this).val();
        if (val == 19) {
            $("#awayteam").html(TeamsHtml19)
        }else if (val == 18) {
            $("#awayteam").html(TeamsHtml18)
        }else if (val == 17) {
            $("#awayteam").html(TeamsHtml17)
        }else if (val == 16) {
            $("#awayteam").html(TeamsHtml16)
        }else if (val == 15) {
            $("#awayteam").html(TeamsHtml15)
        }else if (val == 14) {
            $("#awayteam").html(TeamsHtml14)
        }else if (val == 13) {
            $("#awayteam").html(TeamsHtml13)
        }else if (val == 12) {
            $("#awayteam").html(TeamsHtml12)
          }
      })

})
