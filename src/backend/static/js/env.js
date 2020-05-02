let DOMAIN = window.location.protocol + "//" + window.location.host;

var trajectory1;
var trajectory2;

$(document).ready(function(){
    get_pair();
 });


function update_tragectory(data) {
    console.log("updating trajectories");
    trajectory1 = JSON.parse(data).t1;
    trajectory2 = JSON.parse(data).t2;
    var traj1 = "data:video/mp4;base64,"+trajectory1;
    var traj2 = "data:video/mp4;base64,"+trajectory2;
    $("#traj1").attr("src", traj1);
    $("#traj2").attr("src", traj2);
}

function get_pair() {
    console.log("getting a new pair");
$.get( "/getpair", function(data) {
        update_tragectory(data);
    })
    .fail(function(error) {
        alert( error );
    })
}

function send_preference(pref) {
    console.log("sending pref");
    $.ajax({
    type: "POST",
    url: "/preference",
    //return a json string where t is the trajectory and p is the preference
    data: JSON.stringify({ t1: trajectory1, t2: trajectory2,p: pref }),
    contentType: "application/json; charset=utf-8",
    success: function(data){update_tragectory(data);},
    failure: function(errMsg) {
        get_pair();
    }
});
}

function left_clicked() {
    send_preference("L");
}

function center_clicked() {
    send_preference("N");
}

function right_clicked() {
    send_preference("R");
}