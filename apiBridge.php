<?php
$db = new SQLite3('~/DEV/CV2C/bridge.db');

$action = $_GET['action'] ?? '';

if ($action === 'getTables') {
    $res = $db->query("SELECT id, nom FROM tables");
    $tables = [];
    while ($row = $res->fetchArray(SQLITE3_ASSOC)) {
        $tables[] = $row;
    }
    echo json_encode($tables);
}

elseif ($action === 'getContrats') {
    $table_id = intval($_GET['table_id']);
    $res = $db->query("SELECT id, contrat, joueur FROM contrats WHERE table_id = $table_id");
    $contrats = [];
    while ($row = $res->fetchArray(SQLITE3_ASSOC)) {
        $contrats[] = $row;
    }
    echo json_encode($contrats);
}

elseif ($action === 'getPlis') {
    $contrat_id = intval($_GET['contrat_id']);
    $res = $db->query("SELECT numero, carte_nord, carte_est, carte_sud, carte_ouest, joueur FROM plis WHERE contrat_id = $contrat_id ORDER BY numero ASC");
    $plis = [];
    while ($row = $res->fetchArray(SQLITE3_ASSOC)) {
        $plis[] = $row;
    }
    echo json_encode($plis);
}

else {
    echo json_encode(["error" => "Action inconnue"]);
}

