// $(document).ready(function(){
//     get_pair();
//  });

const filenames = {"cart":"cartpole.png", "hop":"hopper.png", "human":"humanoid.png", "lunar":"lunar_lander.png"};
available_envs = ["cart", "hop", "human", "lunar"];
let img_on_row = 0;

function populate_page(available_envs) {
    for(env of available_envs) {
        filename = filenames[env];
        if(img_on_row < 4){
            img_on_row++;

        } else {
            img_on_row = 0;
            // create new row
        }
    }
    let append_text = "<b>Appended text</b>";
    $("#env_links").append(append_text);
}



function get_envs() {
$.get( "/getenvs", function(data) {
        update_sequences(data);
    })
    .fail(function(error) {
        alert( error );
    })
}