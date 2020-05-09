//creating teams html from array
function createTeamHtml(teamList) {
  team_result = []
  team_result.push("<option value='none' selected disabled hidden>Select Team:</option>")
  for(let i = 0;i < teamList.length;i++){
    var teamHtml = ""
    var value = teamHtml.concat("<option value=", teamList[i].replace(/\s/g, ''), ">", teamList[i], "</option>");
    team_result.push(value);
  }
  return team_result;
}

teams19 = ['New York Empire',
 'San Diego Growlers',
 'Raleigh Flyers',
 'Dallas Roughnecks',
 'Pittsburgh Thunderbirds',
 'Indianapolis AlleyCats',
 'Los Angeles Aviators',
 'Toronto Rush',
 'Chicago Wildfire',
 'DC Breeze',
 'Madison Radicals',
 'Minnesota Wind Chill',
 'Atlanta Hustle',
 'Philadelphia Phoenix',
 'Montreal Royal',
 'Tampa Bay Cannons',
 'San Jose Spiders',
 'Austin Sol',
 'Ottawa Outlaws',
 'Seattle Cascades',
 'Detroit Mechanix'];


teams18 = ['Dallas Roughnecks',
 'Madison Radicals',
 'Toronto Rush',
 'Indianapolis AlleyCats',
 'Los Angeles Aviators',
 'Raleigh Flyers',
 'New York Empire',
 'Minnesota Wind Chill',
 'DC Breeze',
 'Atlanta Hustle',
 'San Diego Growlers',
 'Austin Sol',
 'Montreal Royal',
 'San Jose Spiders',
 'Philadelphia Phoenix',
 'Seattle Cascades',
 'Chicago Wildfire',
 'San Francisco FlameThrowers',
 'Pittsburgh Thunderbirds',
 'Tampa Bay Cannons',
 'Ottawa Outlaws',
 'Nashville Nightwatch',
 'Detroit Mechanix']

 teams17 = ['Dallas Roughnecks',
 'Raleigh Flyers',
 'Madison Radicals',
 'Toronto Rush',
 'San Francisco FlameThrowers',
 'Minnesota Wind Chill',
 'Los Angeles Aviators',
 'DC Breeze',
 'Pittsburgh Thunderbirds',
 'Tampa Bay Cannons',
 'Montreal Royal',
 'San Jose Spiders',
 'San Diego Growlers',
 'Seattle Cascades',
 'New York Empire',
 'Indianapolis AlleyCats',
 'Atlanta Hustle',
 'Austin Sol',
 'Philadelphia Phoenix',
 'Chicago Wildfire',
 'Ottawa Outlaws',
 'Detroit Mechanix',
 'Vancouver Riptide',
 'Nashville Nightwatch']

 teams16 = ['Dallas Roughnecks',
 'Madison Radicals',
 'Toronto Rush',
 'Seattle Cascades',
 'DC Breeze',
 'Pittsburgh Thunderbirds',
 'San Francisco FlameThrowers',
 'Minnesota Wind Chill',
 'Raleigh Flyers',
 'Los Angeles Aviators',
 'Indianapolis AlleyCats',
 'Atlanta Hustle',
 'Austin Sol',
 'New York Empire',
 'Ottawa Outlaws',
 'Vancouver Riptide',
 'Montreal Royal',
 'San Jose Spiders',
 'Tampa Bay Cannons',
 'Charlotte Express',
 'Chicago Wildfire',
 'Detroit Mechanix',
 'Nashville Nightwatch',
 'San Diego Growlers',
 'Cincinnati Revolution',
 'Philadelphia Phoenix']

 teams15 =['Madison Radicals',
 'Toronto Rush',
 'San Jose Spiders',
 'Pittsburgh Thunderbirds',
 'Raleigh Flyers',
 'New York Empire',
 'Tampa Bay Cannons',
 'Atlanta Hustle',
 'Montreal Royal',
 'Seattle Cascades',
 'San Francisco FlameThrowers',
 'Chicago Wildfire',
 'San Diego Growlers',
 'Ottawa Outlaws',
 'Indianapolis AlleyCats',
 'DC Breeze',
 'Minnesota Wind Chill',
 'Los Angeles Aviators',
 'Vancouver Riptide',
 'Cincinnati Revolution',
 'Charlotte Express',
 'Nashville Nightwatch',
 'Rochester Dragons',
 'Philadelphia Phoenix',
 'Detroit Mechanix']

 teams14 = ['San Jose Spiders',
 'Toronto Rush',
 'Madison Radicals',
 'San Francisco FlameThrowers',
 'New York Empire',
 'DC Breeze',
 'Indianapolis AlleyCats',
 'Chicago Wildfire',
 'Minnesota Wind Chill',
 'Vancouver Riptide',
 'Montreal Royal',
 'Cincinnati Revolution',
 'Seattle Cascades',
 'Philadelphia Phoenix',
 'Rochester Dragons',
 'Detroit Mechanix']

 teams13 = ['Toronto Rush',
 'Madison Radicals',
 'Chicago Wildfire',
 'New York Empire',
 'Philadelphia Phoenix',
 'Indianapolis AlleyCats',
 'Rochester Dragons',
 'Detroit Mechanix',
 'Minnesota Wind Chill',
 'Cincinnati Revolution',
 'DC Breeze',
 'New Jersey Hammerheads']

 teams12 = ['Philadelphia Spinners',
 'Indianapolis AlleyCats',
 'Cincinnati Revolution',
 'Connecticut Constitution',
 'Columbus Cranes',
 'Detroit Mechanix',
 'Rhode Island Rampage',
 'Rochester Dragons']

 TeamsHtml19 = createTeamHtml(teams19);
 TeamsHtml18 = createTeamHtml(teams18);
 TeamsHtml17 = createTeamHtml(teams17);
 TeamsHtml16 = createTeamHtml(teams16);
 TeamsHtml15 = createTeamHtml(teams15);
 TeamsHtml14 = createTeamHtml(teams14);
 TeamsHtml13 = createTeamHtml(teams13);
 TeamsHtml12 = createTeamHtml(teams12);
