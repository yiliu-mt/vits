var player;

$(document).ready(function () {
        $(".alert").hide();
});

function download(fileURL) {
  const a = document.createElement('a')
  a.href = fileURL
  a.download = url.split('/').pop()
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}


$('#generate_action').on('click',function() {

    // get speaker selection
    speaker = $('#speaker_selection').val();
    version = $('#version').val();
    // get other info
    pitch = parseFloat($('#pitch_range').val());
    pitch = (pitch - 1.0) * 500
    duration = parseFloat($('#duration_range').val());
    duration = (duration - 1.0) * 500
    energy = parseFloat($('#energy_range').val());
    energy = (energy - 1.0) * 500
    volume = parseFloat($('#volume_range').val());
    volume = volume * 50
    text = $('#tts_text').val();

    // console.log("request:", model, speaker, pitch, duration, energy, text);

    request_data = {
        speaker: speaker,
        pitch_rate: pitch,
        energy_rate: energy,
        speech_rate: duration,
        volume: volume,
        text: text,
        version: version,
    };

    console.log(JSON.stringify(request_data));

    var xhrOverride = new XMLHttpRequest();
    xhrOverride.responseType = 'arraybuffer';


    // reset
    audio_obj = $('#audio_play');
    audio_obj.get(0).src = "";
    audio_obj.get(0).load();

    $(".alert").hide();
    $('#download_action').unbind();
    $("#download_action").hide();

    // send out request
    $.ajax({
     url:'/generate',
     type: "POST",
     data: JSON.stringify(request_data),
    contentType: 'application/json',
        xhr: function() {
        return xhrOverride;
    }
    }).done(function (data) {
        $('#audio_play').attr('hidden', false);
        // playArrayBuffer(data);
        const blob = new Blob([data], { type: "audio/wav" });
        console.log(blob);
        url = window.URL.createObjectURL(blob);
        audio_obj = $('#audio_play');

        console.log(url);
        audio_obj.get(0).src = url;
        audio_obj.get(0).load();

        $("#success_alert").hide().show();

        $('#download_action').on('click', function(event) {
              event.preventDefault(); // To prevent following the link (optional)
            download(url);
        });

        $("#download_action").hide().show();

    }).fail(function(jqXHR, textStatus, errorThrown){
        console.log(jqXHR);
        console.log(textStatus);
        console.log(errorThrown);

        audio_obj = $('#audio_play');
        audio_obj.get(0).src = "";
        audio_obj.get(0).load();

        $("#fail_alert").hide().show();
    });

});


