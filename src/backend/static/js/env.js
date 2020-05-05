let DOMAIN = window.location.protocol + "//" + window.location.host;

let sequence1;
let sequence2;
let last_response = Date.now();  // unix timestamp in milliseconds

$(document).ready(function(){
    get_pair();
    console.log(last_response);
 });


function update_sequences(data) {
    console.log("updating sequences");

    data = JSON.parse(data);
    if(typeof data.error !== 'undefined') {
        alert(data.error);
    } else {
        sequence1 = data.seq1;
        sequence2 = data.seq2;
        var traj1 = "data:video/mp4;base64," + sequence1.vid;
        var traj2 = "data:video/mp4;base64," + sequence2.vid;
        $("#trajL").attr("src", traj1);
        $("#trajR").attr("src", traj2);
    }
}

function get_pair() {
    console.log("getting a new pair");
$.get( "/getpair?env="+env, function(data) {
        update_sequences(data);
    })
    .fail(function(error) {
        alert( "ERROR" );
    })
}

function no_response_checker() {
    console.log("Beginning no-response-checker...");
    setTimeout(function(){
        diff = Date.now() - last_response;
        console.log("Its been some time, diff: "+diff);
        if(diff > 1999) {
            // alert("The server took too long to respond!");
        }
    }, 2000);
}

function send_preference(pref) {
    last_response = Date.now();
    no_response_checker();
    $.ajax({
    type: "POST",
    url: "/preference",
    //return a json string where t is the sequence and p is the preference
    data: JSON.stringify({ env: env, seq1: sequence1.sopairs, seq2: sequence2.sopairs, p: pref }),
    contentType: "application/json; charset=utf-8",
    success: function(data){
        last_response = Date.now();
        update_sequences(data);},
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