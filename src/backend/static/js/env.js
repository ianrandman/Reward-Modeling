let DOMAIN = window.location.protocol + "//" + window.location.host;

var date = new Date();
let sequence1;
let sequence2;
let last_response = date.getTime();  // unix timestamp in milliseconds

$(document).ready(function(){
    get_pair();
    console.log(last_response);
 });


function update_sequences(data) {
    console.log("updating sequences");
    sequence1 = JSON.parse(data).seq1;
    sequence2 = JSON.parse(data).seq2;
    var traj1 = "data:video/mp4;base64,"+sequence1.vid;
    var traj2 = "data:video/mp4;base64,"+sequence2.vid;
    $("#trajL").attr("src", traj1);
    $("#trajR").attr("src", traj2);
}

function get_pair() {
    console.log("getting a new pair");
$.get( "/getpair?env="+env, function(data) {
        update_sequences(data);
    })
    .fail(function(error) {
        alert( error );
    })
}

function no_response_checker() {
    
}

function send_preference(pref) {
    no_response_checker();
    $.ajax({
    type: "POST",
    url: "/preference",
    //return a json string where t is the sequence and p is the preference
    data: JSON.stringify({ env: env, seq1: sequence1.sopairs, seq2: sequence2.sopairs, p: pref }),
    contentType: "application/json; charset=utf-8",
    success: function(data){update_sequences(data);},
    failure: function(errMsg) {
        console.log("failed to send preference");
        get_pair();
    }
});
}

function left_clicked() {
    send_preference(0);
}

function center_clicked() {
    send_preference(0.5);
}

function right_clicked() {
    send_preference(1);
}