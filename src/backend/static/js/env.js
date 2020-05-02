let DOMAIN = window.location.protocol + "//" + window.location.host;

var trajectory;

$(document).ready(function(){
    get_pair();
 });


function update_tragectory(data) {
    trajectory = JSON.parse(data).data;
    $("#diagnostics").text(trajectory);
}

function get_pair() {
$.get( "/getpair", function(data) {
        update_tragectory(data);
    })
    .fail(function(error) {
        alert( error );
    })
}

function send_preference(pref) {
    $.ajax({
    type: "POST",
    url: "/preference",
    //return a json string where t is the trajectory and p is the preference
    data: JSON.stringify({ t: trajectory, p: pref }),
    contentType: "application/json; charset=utf-8",
    dataType: "json",
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