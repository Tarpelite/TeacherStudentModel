import re
AUS_PRIMARY_LABEL = ['Booking', 'CargoRelease', 'ContainerHandling', 'Customs', 'Documentation', 'General', 'Invoice/Payment', 'None', 'Rate',
                    'Report', 'Unclassified']
AUS_SECONDARY_LABEL = ['Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel', 'Booking-Amendment', 'Booking-FirmUp',
                    'Booking-Request-Attachment', 'CargoRelease', 'CargoRelease-PinRelease', 'CargoRelease-ReleaseInstruction', 'ContainerHandling',
                    'ContainerHandling-EmptyRelease', 'ContainerHandling-Reuse', 'ContainerHandling-EmptyRestitution', 'ContainerHandling-DND',
                    'Customs', 'Documentation-ConsignmentNote-Attachment', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation',
                    'Documentation-BL-Amendment', 'Documentation-BL', 'Documentation-SI-Attachment', 'Documentation-ArrivalNotice',
                    'General-Service/Vessel', 'Invoice/Payment', 'Invoice/Payment-Payment', 'Invoice/Payment-Invoice', 'None', 'Rate', 'Report', 'Unclassified']

UKD_PRIMARY_LABEL = ['None', 'Report', 'Booking', 'CargoRelease', 'ContainerHandling', 'Customs', 'Documentation', 'General', 'Invoice/Payment',
                'Rate', 'Transportation', 'UCR', 'Unclassified']
UKD_SECONDARY_LABEL = ['None', 'Report', 'Booking', 'Booking-Request', 'Booking-DGN', 'Booking-Confirmation', 'Booking-Cancel', 'Booking-Amendment',
                'Booking-VGM', 'Booking-Request-Attachment', 'CargoRelease-PinRelease', 'CargoRelease-ReleaseInstruction', 'CargoRelease-PinExtend',
                'ContainerHandling', 'ContainerHandling-BookinRequest', 'ContainerHandling-EmptyRestitution', 'ContainerHandling-DND', 'Customs',
                'Customs-SAD', 'Documentation-SI', 'Documentation-BL-Release', 'Documentation-BL-Confirmation', 'Documentation-BL-Amendment',
                'Documentation-BL', 'Documentation-SI-Attachment', 'General-Service/Vessel', 'Invoice/Payment', 'Invoice/Payment-Payment',
                'Invoice/Payment-Invoice', 'Rate', 'Transportation', 'Transportation-DeliveryNote', 'Transportation-Amendment', 'Transportation-Request',
                'UCR', 'Unclassified']

def convert(score_file, data_name, threshold=0.5):
    wr = open(score_file + '.label', 'w', encoding='utf-8')
    labels = AUS_SECONDARY_LABEL
    if data_name == 'ukd':
        labels = UKD_SECONDARY_LABEL
    reader = open(score_file, 'r', encoding='utf-8')
    for line in reader:
        if len(line.split('\t')) < 2:
            continue
        scores, _ = line.split('\t')
        scores = eval(re.sub(' +', ',', scores))
        scores = scores[:len(labels)]
        one_labels = []
        if max(scores) > threshold:
            for i, score in enumerate(scores):
                if score > threshold:
                    one_labels.append(labels[i])
        else:
            one_labels.append(labels[scores.index(max(scores))])
        wr.write(';'.join(one_labels))
        wr.write('\n')
    wr.close()


if __name__ == '__main__':
    convert(r"F:\tmp\oocl_aus_base_512_cls1_epoch10\results_epoch_0.txt", "aus")
