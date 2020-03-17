// var cd = $('#chart_data').val();
// console.log("here is chart data: ", plot_data1);


$('.input-group.date').datepicker({format: "yyyy-mm-dd"});

//  // Material Select Initialization
// $(document).ready(function() {
//     $('.mdb-select').materialSelect();
//     });
 

(function ($) {
  $('.spinner0 .btn:first-of-type').on('click', function() {
    $('.spinner0 input').val( parseFloat($('.spinner0 input').val(),100) + 0.5);
  });
  $('.spinner0 .btn:last-of-type').on('click', function() {
    $('.spinner0 input').val( parseFloat($('.spinner0 input').val(), 0) - 0.5);
  });
})(jQuery);

(function ($) {
    $('.spinner1 .btn:first-of-type').on('click', function() {
      $('.spinner1 input').val( parseFloat($('.spinner1 input').val(),100) + 0.5);
    });
    $('.spinner1 .btn:last-of-type').on('click', function() {
      $('.spinner1 input').val( parseFloat($('.spinner1 input').val(), 0) - 0.5);
    });
  })(jQuery);

  (function ($) {
    $('.spinner2 .btn:first-of-type').on('click', function() {
      $('.spinner2 input').val( parseInt($('.spinner2 input').val(),10) + 1+"%");
    });
    $('.spinner2 .btn:last-of-type').on('click', function() {
      $('.spinner2 input').val( parseInt($('.spinner2 input').val(), 10) - 1+"%");
    });
  })(jQuery);

  (function ($) {
    $('.spinner3 .btn:first-of-type').on('click', function() {
      $('.spinner3 input').val( parseInt($('.spinner3 input').val(),10) + 1+"$");
    });
    $('.spinner3 .btn:last-of-type').on('click', function() {
      $('.spinner3 input').val( parseInt($('.spinner3 input').val(), 10) - 1+"$");
    });
  })(jQuery);


//   var tt=document.getElementById("chart_data1").getAttribute("name")
//   var plotdata1 = JSON.parse(tt)
//   var plotdata2 = JSON.parse(document.getElementById("chart_data2").getAttribute("name"))
//   console.log("here is chart data: ", plotdata1);


if(document.getElementById("chart_data1").getAttribute("name")!=null)
{
    Highcharts.chart('highcharts-container', {

        title: {
            text: 'Equity Curve'
        },

        // subtitle: {
        //     text: 'Source: thesolarfoundation.com'
        // },

        yAxis: {
            title: {
                text: 'Dollars'
            }
        },

        // xAxis: {
        //     accessibility: {
        //         rangeDescription: 'Year'
        //     }
        // },

        xAxis: {
            type: 'datetime',
            dateTimeLabelFormats: {
            day: '%Y %b %d'    //ex- 01 Jan 2016
            },
            accessibility: {
                rangeDescription: 'Year'
            }
        },

        legend: {
            layout: 'vertical',
            align: 'right',
            verticalAlign: 'middle'
        },

        plotOptions: {
            series: {
                label: {
                    connectorAllowed: false
                },
                pointStart: 1
            }
        },

        series: [{
            name: 'Strategy',
            data: JSON.parse(document.getElementById("chart_data1").getAttribute("name"))
        },
        {
            name: 'Benchmark',
            data: JSON.parse(document.getElementById("chart_data2").getAttribute("name"))
        }],
        // series: cd,
        exporting: { enabled: false }
        // responsive: {
        //     rules: [{
        //         condition: {
        //             maxWidth: 500
        //         },
        //         chartOptions: {
        //             legend: {
        //                 layout: 'horizontal',
        //                 align: 'center',
        //                 verticalAlign: 'bottom'
        //             }
        //         }
        //     }]
        // }

    });
}


function hypothesis_engine_Run() {

    var GRAMMAR_FILE = $('#GRAMMAR_FILE').val();
    var FITNESS_FUNCTION = $('#FITNESS_FUNCTION').val();
    var CROSSOVER = $('#CROSSOVER').val();
    var MAX_INIT_TREE_DEPTH = $('#MAX_INIT_TREE_DEPTH').val();
    var MIN_INIT_TREE_DEPTH = $('#MIN_INIT_TREE_DEPTH').val();
    var INIT_GENOME_LENGTH = $('#INIT_GENOME_LENGTH').val();
    var INTERACTION_PROBABILITY = $('#INTERACTION_PROBABILITY').val();
    var MAX_TREE_DEPTH = $('#MAX_TREE_DEPTH').val();
    var POPULATION_SIZE = $('#POPULATION_SIZE').val();
    var SELECTION_PROPORTION = $('#SELECTION_PROPORTION').val();
    var GENERATIONS = $('#GENERATIONS').val();
    var GENERATION_SIZE = $('#GENERATION_SIZE').val();
    var ELITE_SIZE = $('#ELITE_SIZE').val();
    var CROSSOVER_PROBABILITY = $('#CROSSOVER_PROBABILITY').val();
    var MUTATION_EVENTS = $('#MUTATION_EVENTS').val();
    var MUTATION_PROBABILITY = $('#MUTATION_PROBABILITY').val();
    var TOURNAMENT_SIZE = $('#TOURNAMENT_SIZE').val();

    // $('.loading-wrapper').show();
    jQuery.ajax({
        url: 'http://127.0.0.1:5000/hypothesis_engine',
        type: 'post',
        data: {GRAMMAR_FILE: GRAMMAR_FILE, FITNESS_FUNCTION:FITNESS_FUNCTION, CROSSOVER:CROSSOVER, MAX_INIT_TREE_DEPTH:MAX_INIT_TREE_DEPTH,
            MIN_INIT_TREE_DEPTH:MIN_INIT_TREE_DEPTH,INIT_GENOME_LENGTH:INIT_GENOME_LENGTH, INTERACTION_PROBABILITY:INTERACTION_PROBABILITY,
            MAX_TREE_DEPTH:MAX_TREE_DEPTH,POPULATION_SIZE:POPULATION_SIZE,SELECTION_PROPORTION:SELECTION_PROPORTION,
            GENERATIONS:GENERATIONS,GENERATION_SIZE:GENERATION_SIZE,ELITE_SIZE:ELITE_SIZE,CROSSOVER_PROBABILITY:CROSSOVER_PROBABILITY,
            MUTATION_EVENTS:MUTATION_EVENTS, MUTATION_PROBABILITY:MUTATION_PROBABILITY,TOURNAMENT_SIZE:TOURNAMENT_SIZE
        },

        success: function(data){
            console.log("success");
            // var outData = data.split(';');
            $('#output_tag').empty();
            $('#output_tag').html(data);

            // for (var i =0; i<outData.length ; i++)
            // {
            //     console.log(outData[i]);
            // }


            // var table = data.split(';')[0];
            // var clusters = JSON.parse(data.split(';')[1]);

            // console.log(clusters[0][2]);
            // $('#result_table').empty();
            // $('#result_table').html(table);
            // $('#result_title').html("Results");
            // setTimeout(function() {
            //     $('.loading-wrapper').hide();
            // }, 500);

            // $('#chart-container').empty();
            
        }
    });
}